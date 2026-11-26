"""Uploads to S3: use a private ACL when the bucket allows it, and fall back to none when it does not.

A fake S3 client stands in for AWS, so these tests need no account, network or boto3 client.
"""
import asyncio
import io
import logging
import threading

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException

from app.services import file_management
from app.services.file_management import FileStorageService

LOGGER = "app.services.file_management"


def client_error(code, message="refused"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "PutObject")


class FakeS3:
    """Records every put_object and presigned link. The *_error fields decide what AWS refuses."""

    def __init__(self, with_acl=None, without_acl=None):
        self.calls = []
        self.presigned = []
        self.with_acl = with_acl  # error raised when an ACL is sent, or None to accept
        self.without_acl = without_acl  # error raised when it is not sent
        self.before_reply = None  # optional hook, to line up concurrent uploads

    def put_object(self, **kwargs):
        body = kwargs.get("Body")
        # Like the real client, read a stream while sending: that uses it up
        uploaded = body.read() if hasattr(body, "read") else body
        self.calls.append({**kwargs, "uploaded": uploaded})
        if self.before_reply:
            self.before_reply()
        error = self.with_acl if "ACL" in kwargs else self.without_acl
        if error is not None:
            raise error

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None, **_):
        self.presigned.append((operation, Params))
        return f"https://bucket.example/{Params['Key']}?signed"


@pytest.fixture()
def make_storage(monkeypatch):
    def fake_init(self):
        self.s3_client = object()
        self.bucket_name = "test-bucket"
        self.cdn_endpoint = None
        self.upload_expiration = 3600
        self.download_expiration = 86400
        self.max_file_size = 10 * 1024 * 1024
        self.allowed_mime_types = ["application/pdf"]

    monkeypatch.setattr(FileStorageService, "_init_s3_client", fake_init)
    # The "skip the ACL for a while" memory is shared by all uploads: start every test clean
    monkeypatch.setattr(FileStorageService, "_acl_skip_until", 0.0)

    def build(**fake_rules):
        service = FileStorageService()
        service.s3_client = FakeS3(**fake_rules)
        return service

    return build


def upload(storage):
    return storage.upload_file_bytes(b"%PDF-1.4", "batch_reports/report.pdf", "application/pdf")


def acl_flags(storage):
    return ["ACL" in call for call in storage.s3_client.calls]


def warnings(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


# --- the fallback ---------------------------------------------------------------------------

def test_private_acl_is_used_when_the_bucket_allows_it(make_storage, caplog):
    storage = make_storage()
    upload(storage)
    assert acl_flags(storage) == [True]
    assert storage.s3_client.calls[0]["ACL"] == "private"
    assert warnings(caplog) == []


@pytest.mark.parametrize("code", ["AccessControlListNotSupported", "AccessDenied"])
def test_upload_succeeds_without_the_acl_when_the_bucket_rejects_it(make_storage, code, caplog):
    storage = make_storage(with_acl=client_error(code))
    assert upload(storage) == "batch_reports/report.pdf"
    assert acl_flags(storage) == [True, False]  # tried with, then without
    (warning,) = warnings(caplog)
    assert code in warning.getMessage()


def test_the_warning_does_not_claim_the_files_are_private(make_storage, caplog):
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    upload(storage)
    message = warnings(caplog)[0].getMessage()
    assert "stay private" not in message
    assert "Block Public Access" in message  # points at what really decides it


def test_every_attempt_is_encrypted_and_aimed_at_the_same_file(make_storage):
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    upload(storage)
    for call in storage.s3_client.calls:
        assert call["ServerSideEncryption"] == "AES256"
        assert call["Bucket"] == "test-bucket"
        assert call["Key"] == "batch_reports/report.pdf"
        assert call["uploaded"] == b"%PDF-1.4"


def test_later_uploads_skip_the_acl_instead_of_failing_first(make_storage, caplog):
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    upload(storage)
    storage.s3_client.calls.clear()
    upload(storage)
    upload(storage)
    assert acl_flags(storage) == [False, False]
    assert len(warnings(caplog)) == 1  # warned once, not per upload


def test_the_acl_is_tried_again_after_the_skip_period(make_storage, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(file_management.time, "monotonic", lambda: clock[0])
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    upload(storage)

    storage.s3_client.with_acl = None  # the bucket is fixed meanwhile
    storage.s3_client.calls.clear()
    upload(storage)
    assert acl_flags(storage) == [False]  # still inside the skip period

    clock[0] += FileStorageService._ACL_SKIP_SECONDS + 1
    storage.s3_client.calls.clear()
    upload(storage)
    assert acl_flags(storage) == [True]  # tried again, and it works now


def test_direct_file_upload_uses_the_same_fallback(make_storage):
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    path = storage.direct_file_upload("policy.pdf", b"%PDF-1.4", "application/pdf")
    assert path.endswith(".pdf")
    assert acl_flags(storage) == [True, False]


# --- a real permission problem must not be blamed on the ACL --------------------------------

def test_a_real_permission_problem_is_reported_and_leaves_the_acl_alone(make_storage, caplog):
    # Without s3:PutObject both attempts fail: the error of the second attempt must surface
    denied = client_error("AccessDenied", "not authorized to perform: s3:PutObject")
    storage = make_storage(with_acl=denied, without_acl=denied)
    with pytest.raises(HTTPException) as caught:
        upload(storage)
    assert caught.value.status_code == 500
    assert "s3:PutObject" in caught.value.detail
    assert acl_flags(storage) == [True, False]
    assert warnings(caplog) == []  # the ACL was not the cause, so nothing blames it

    # The permission is added afterwards: the very next upload still tries the ACL first
    storage.s3_client.with_acl = storage.s3_client.without_acl = None
    storage.s3_client.calls.clear()
    upload(storage)
    assert acl_flags(storage) == [True]


@pytest.mark.parametrize("code", ["NoSuchBucket", "InternalError", "SlowDown", "InvalidAccessKeyId"])
def test_other_errors_are_not_retried_without_the_acl(make_storage, code):
    storage = make_storage(with_acl=client_error(code))
    with pytest.raises(HTTPException):
        upload(storage)
    assert acl_flags(storage) == [True]  # one attempt only


# --- file data that is a stream --------------------------------------------------------------

def test_a_streamed_file_is_rewound_so_the_retry_uploads_all_of_it(make_storage):
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    storage._put_object(Bucket="test-bucket", Key="k", Body=io.BytesIO(b"whole file"))
    assert [call["uploaded"] for call in storage.s3_client.calls] == [b"whole file", b"whole file"]


class UnseekableStream:
    def read(self):
        return b"data"

    def seek(self, _):
        raise io.UnsupportedOperation("not seekable")


def test_a_stream_that_cannot_be_rewound_is_not_retried(make_storage):
    # A retry would upload an empty file and report success, so the real error is raised instead
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    with pytest.raises(ClientError):
        storage._put_object(Bucket="test-bucket", Key="k", Body=UnseekableStream())
    assert acl_flags(storage) == [True]


# --- several uploads at once ------------------------------------------------------------------

def test_simultaneous_first_uploads_warn_once(make_storage, caplog):
    workers = 6
    storage = make_storage(with_acl=client_error("AccessControlListNotSupported"))
    barrier = threading.Barrier(workers)
    # Every thread has made its ACL attempt before any of them gets its answer
    storage.s3_client.before_reply = lambda: barrier.wait(timeout=5) if storage.s3_client.with_acl else None
    first_attempts_done = []

    original = storage.s3_client.put_object

    def put_object(**kwargs):
        try:
            return original(**kwargs)
        finally:
            if "ACL" in kwargs:
                first_attempts_done.append(1)

    storage.s3_client.put_object = put_object
    threads = [threading.Thread(target=upload, args=(storage,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(first_attempts_done) == workers
    assert len(warnings(caplog)) == 1


# --- presigned upload links -------------------------------------------------------------------

def test_presigned_upload_link_is_encrypted_and_signs_no_acl(make_storage):
    storage = make_storage()
    result = asyncio.run(storage.generate_upload_url("policy.pdf", "application/pdf"))
    ((operation, params),) = storage.s3_client.presigned
    assert operation == "put_object"
    assert params["ServerSideEncryption"] == "AES256"
    assert "ACL" not in params  # a signed link cannot fall back, so it must not demand an ACL
    assert result.upload_url.startswith("https://bucket.example/")
