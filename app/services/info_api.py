from fastapi import APIRouter, Depends, Header, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.core.admin_auth import is_admin
from app.db.base_db import mongo_client
from pydantic import BaseModel, Field

router = APIRouter()


def require_admin(
    x_admin_key: Optional[str] = Header(None, description="The operator key from ADMIN_API_KEY")
) -> None:
    """
    Changing the reference data is for the operator only. The shared API key is also held by the
    website, which adds it to every request it passes on, so on its own it would let any visitor
    create, overwrite or delete a country's records.
    """
    if not is_admin(x_admin_key):
        raise HTTPException(
            status_code=403,
            detail="Changing reference data requires a valid X-Admin-Key header"
        )



class LawsDataResponse(BaseModel):
    country: str = Field(..., description="The name of the country")
    yearLegislationEnacted:  str = Field(
        ..., description="The year the legislation was enacted")
    malaboConventionStatus: str = Field(
        ..., description="The status of the Malabo Convention")
    # name: str = Field(..., description="The name of the law")
    dataProtectionLegislationStatus: str = Field(
        ..., description="The status of the data protection legislation")
    
    links: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of links related to the law"
    )
    isoAlpha2: str = Field(
        ..., description="ISO2 code for the country representation"
    )
    flagEmoji: Optional[str] = Field(
        None, description="Emoji representation of the country's flag"
    )
    
    


@router.get("/policy_laws", response_model=List[LawsDataResponse])
async def get_policy_laws(page: int = 1, limit: int = 55):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        skip = (page - 1) * limit
        laws = list(collection.find({}, {"_id": 0}).skip(skip).limit(limit))

        if not laws:
            raise HTTPException(
                status_code=404, detail="No policy laws found.")
        print(f"Retrieved {laws} policy laws from the database.")
        laws = [LawsDataResponse(**law) for law in laws]
        return laws
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving policy laws: {str(e)}")


class DigitalIDDataResponse(BaseModel):
    country: str = Field(..., description="The name of the country")
    status:str = Field(..., description="The status of the digital ID system")
    flagEmoji: Optional[str] = Field(
        None, description="Emoji representation of the country's flag")
    year_launched: Optional[str] = Field(
        None, description="The year the digital ID system was launched")
    isoAlpha2: Optional[str] = Field(
        None, description="The ISO Alpha-2 code of the country")

    features: Optional[List[str]] = Field(
        None, description="List of features of the digital ID system")
    coverage: Optional[str] = Field(
        None, description="Coverage of the digital ID system")
    authority: Optional[str] = Field(
        None, description="Authority responsible for the digital ID system")
    description: Optional[str] = Field(
        None, description="Description of the digital ID system")
    links: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of links related to the digital ID system")


@router.get("/digital_id", response_model=List[DigitalIDDataResponse])
async def get_digital_id(page: int = 1, limit: int = 55):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        skip = (page - 1) * limit
        digital_ids = list(collection.find(
            {}, {"_id": 0}).skip(skip).limit(limit))

        if not digital_ids:
            raise HTTPException(
                status_code=404, detail="No digital ID data found.")

        digital_ids = [DigitalIDDataResponse(
            **digital_id) for digital_id in digital_ids]
        return digital_ids
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving digital ID data: {str(e)}")


class LawsDataFileResponse(BaseModel):
    country: str = Field(..., description="The name of the country")
    isoAlpha2: Optional[str] = Field(
        None, description="The ISO Alpha-2 code of the country")
    link: Optional[str] = Field(
        None, description="Link to the data protection law")


@router.get("/policy_laws/files", response_model=List[LawsDataFileResponse])
async def get_policy_laws_files(page: int = 1, limit: int =
                                55):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        skip = (page - 1) * limit
        laws = list(collection.find({
              "link": {"$exists": True, "$ne": ""}
            }, {"_id": 0}).skip(skip).limit(limit))

        # Extract only the required fields for the file response
        law_files = []
        for law in laws:
            law_file = {
                "country": law.get("country", ""),
                "isoAlpha2": law.get("isoAlpha2", ""),
                "link": law.get("link", "")
            }
            law_files.append(LawsDataFileResponse(**law_file))
        
        return law_files
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving policy laws files: {str(e)}")


class DigitalIDFileResponse(BaseModel):
    country: str = Field(..., description="The name of the country")
    isoAlpha2: Optional[str] = Field(
        None, description="The ISO Alpha-2 code of the country")
    link: Optional[str] = Field(
        None, description="Link to the digital ID data")


@router.get("/digital_id/files", response_model=List[DigitalIDFileResponse])
async def get_digital_id_files(page: int = 1, limit: int =55):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        skip = (page - 1) * limit
        digital_ids = list(collection.find(
            {}, {"_id": 0}).skip(skip).limit(limit))

        # Extract only the required fields for the file response
        digital_id_files = []
        for digital_id in digital_ids:
            digital_id_file = {
                "country": digital_id.get("country", ""),
                "isoAlpha2": digital_id.get("isoAlpha2", ""),
                "link": digital_id.get("link", "")
            }
            digital_id_files.append(DigitalIDFileResponse(**digital_id_file))
        
        return digital_id_files
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving digital ID files: {str(e)}")


@router.get("/policy_laws/country/{country_code}", response_model=LawsDataResponse)
async def get_policy_laws_by_country(country_code: str):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        law = collection.find_one({"isoAlpha2": country_code}, {"_id": 0})

        if not law:
            raise HTTPException(
                status_code=404, detail="No policy law found for the specified country.")

        return LawsDataResponse(**law)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving policy laws by country: {str(e)}")


@router.get("/digital_id/country/{country_code}", response_model=DigitalIDDataResponse)
async def get_digital_id_by_country(country_code: str):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        digital_id = collection.find_one(
            {"isoAlpha2": country_code}, {"_id": 0})

        if not digital_id:
            raise HTTPException(
                status_code=404, detail="No digital ID data found for the specified country.")
        return DigitalIDDataResponse(**digital_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving digital ID by country: {str(e)}")


@router.patch("/policy_laws/country/{country_code}", response_model=LawsDataResponse, dependencies=[Depends(require_admin)])
async def update_policy_laws_by_country(country_code: str, law_data: LawsDataResponse):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        result = collection.update_one({"isoAlpha2": country_code}, {
                                       "$set": law_data.dict()})
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404, detail="No policy law found for the specified country.")
        updated_law = collection.find_one(
            {"isoAlpha2": country_code}, {"_id": 0})
        return LawsDataResponse(**updated_law)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating policy laws by country: {str(e)}")


@router.patch("/digital_id/country/{country_code}", response_model=DigitalIDDataResponse, dependencies=[Depends(require_admin)])
async def update_digital_id_by_country(country_code: str, digital_id_data: DigitalIDDataResponse):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        result = collection.update_one({"isoAlpha2": country_code}, {
                                       "$set": digital_id_data.dict()})
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404, detail="No digital ID data found for the specified country.")
        updated_digital_id = collection.find_one(
            {"isoAlpha2": country_code}, {"_id": 0})
        return DigitalIDDataResponse(**updated_digital_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating digital ID by country: {str(e)}")


@router.delete("/policy_laws/country/{country_code}", response_model=dict, dependencies=[Depends(require_admin)])
async def delete_policy_laws_by_country(country_code: str):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        result = collection.delete_one({"isoAlpha2": country_code})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404, detail="No policy law found for the specified country.")
        return {"message": "Policy law deleted successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting policy laws by country: {str(e)}")


@router.delete("/digital_id/country/{country_code}", response_model=dict, dependencies=[Depends(require_admin)])
async def delete_digital_id_by_country(country_code: str):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        result = collection.delete_one({"isoAlpha2": country_code})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404, detail="No digital ID data found for the specified country.")
        return {"message": "Digital ID data deleted successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting digital ID by country: {str(e)}")


@router.post("/policy_laws", response_model=LawsDataResponse, dependencies=[Depends(require_admin)])
async def create_policy_law(law_data: LawsDataResponse):
    try:
        collection = mongo_client.get_collection('policy_law_docs')
        existing_law = collection.find_one(
            {"isoAlpha2": law_data.isoAlpha2}, {"_id": 0})
        if existing_law:
            raise HTTPException(
                status_code=400, detail="Policy law for this country already exists.")

        result = collection.insert_one(law_data.dict())
        new_law = collection.find_one({"_id": result.inserted_id}, {"_id": 0})
        return LawsDataResponse(**new_law)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating policy law: {str(e)}")


@router.post("/digital_id", response_model=DigitalIDDataResponse, dependencies=[Depends(require_admin)])
async def create_digital_id(digital_id_data: DigitalIDDataResponse):
    try:
        collection = mongo_client.get_collection('digital_ids_docs')
        existing_digital_id = collection.find_one(
            {"isoAlpha2": digital_id_data.isoAlpha2}, {"_id": 0})
        if existing_digital_id:
            raise HTTPException(
                status_code=400, detail="Digital ID data for this country already exists.")

        result = collection.insert_one(digital_id_data.dict())
        new_digital_id = collection.find_one(
            {"_id": result.inserted_id}, {"_id": 0})
        return DigitalIDDataResponse(**new_digital_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating digital ID: {str(e)}")
