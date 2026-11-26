"""Language detection and the web-search triggers match whole words, not parts of words.

Substring matching took English questions for French ("ces" in "services", "sur" in "ensure"),
so they were answered in French, and started a web search for most questions ("now" in "know",
"count" in "country"). No model or network is used here.
"""
import pytest

from app.services.query_handler import DynamicQueryHandler, matched_terms


@pytest.fixture(scope="module")
def handler():
    return DynamicQueryHandler()  # components load lazily, so nothing is contacted


ENGLISH = [
    "How do digital ID services ensure access for everyone?",
    "Compare the requirements for commercial data processors",
    "What does Kenya's data protection law say about consent?",
    "What does the AU Malabo Convention say about data protection?",
    "Is there a public comment period for the draft law?",
    "Which countries in South Africa's region have a national ID?",
    "What is the process to obtain a digital ID in Côte d'Ivoire?",
    "Describe the design of non-binding data guidelines",
]

FRENCH = [
    "Quelle est la loi de protection des données du Cameroun?",
    "Comment obtenir une carte d'identité au Kenya?",
    "Les lois sur la protection des données au Sénégal",
    "Bonjour, je veux savoir les sanctions",
    "J’ai une question sur la vie privée",  # typographic apostrophe
    "Pourquoi tu réponds en anglais?",
]


@pytest.mark.parametrize("question", ENGLISH)
def test_english_questions_are_english(handler, question):
    assert handler._detect_query_language(question) == "en"


@pytest.mark.parametrize("question", FRENCH)
def test_french_questions_are_french(handler, question):
    assert handler._detect_query_language(question) == "fr"


ENOUGH_DOCUMENTS = [{"content": "x"}] * 5
FEW_DOCUMENTS = [{"content": "x"}] * 2


@pytest.mark.parametrize("question", [
    "I want to know the penalties for a data breach in Nigeria",
    "Which country has the strictest consent rules?",
    "How should an account holder configure consent?",
    "What is the acknowledged standard for biometric data?",
])
def test_ordinary_questions_do_not_start_a_web_search(handler, question):
    assert handler._should_use_web_search(question, FEW_DOCUMENTS, True) is False


@pytest.mark.parametrize("question", [
    "What are the latest changes to Kenya's data law?",
    "Has Rwanda updated its digital ID rules recently?",
    "What is the current status of the Malabo Convention?",
    "Who enforces the law now?",
    "Quelles sont les dernières mises à jour au Bénin?",
])
def test_questions_about_recent_events_start_a_web_search(handler, question):
    assert handler._should_use_web_search(question, ENOUGH_DOCUMENTS, True) is True


def test_fact_questions_search_the_web_only_when_few_documents_were_found(handler):
    question = "How many countries have a data protection authority?"
    assert handler._should_use_web_search(question, FEW_DOCUMENTS, True) is True
    assert handler._should_use_web_search(question, ENOUGH_DOCUMENTS, True) is False


def test_web_search_stays_off_when_disabled_or_needed_for_too_few_documents(handler):
    assert handler._should_use_web_search("latest news", ENOUGH_DOCUMENTS, False) is False
    assert handler._should_use_web_search("anything", [], True) is True  # fewer than 2 documents


def test_matched_terms_finds_whole_words_and_phrases_only():
    assert matched_terms("I know the country count", ["now", "count"]) == {"count"}
    assert matched_terms("Combien de pays?", ["combien", "combien de"]) == {"combien de"}
    assert matched_terms("Anything", []) == set()


@pytest.mark.parametrize("language", ["en", "fr"])
def test_the_out_of_scope_message_is_a_real_markdown_list(language):
    import markdown

    message = DynamicQueryHandler._get_out_of_scope_message(None, language)
    assert markdown.markdown(message).count("<li>") == 6
    assert "•" not in message and "data-protection" not in message
