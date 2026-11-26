from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from app.core.admin_auth import is_admin
from app.core.config import settings
from app.services.prompt_manager import get_prompt_manager
from app.services.llm_factory import LLMFactory
from app.services.vector_store import get_vector_stores_v2
from app.db.base_db import mongo_client


router = APIRouter(prefix="/api/admin", tags=["admin"])


# Authentication dependency
def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, description="The operator key from ADMIN_API_KEY")
):
    """
    Every route here is for the operator only. The shared API_KEY is not enough: the website holds
    it too. While ADMIN_API_KEY is unset (or equal to API_KEY), all of these routes are off.
    """
    if not is_admin(x_admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return x_admin_key



class PromptOverride(BaseModel):
    prompt_key: str = Field(..., description="Prompt identifier")
    prompt_text: str = Field(..., description="New prompt text")
    use_case: Optional[str] = Field(None, description="Optional use case scope")
    updated_by: str = Field(default="admin", description="User making the change")


class PromptResponse(BaseModel):
    prompt_key: str
    prompt_text: str
    use_case: Optional[str]
    updated_at: str
    updated_by: str


class TabConfigResponse(BaseModel):
    tab_name: str
    display_name: str
    description: str
    enabled: bool
    metadata_filters: List[str]
    priority: int


class HealthCheckResponse(BaseModel):
    status: str
    components: Dict[str, str]
    configuration: Dict[str, Any]
    warnings: List[str]


@router.get("/prompts", response_model=Dict[str, Any])
async def list_prompts(admin_key: str = Depends(verify_admin_key)):
    try:
        prompts_config = settings.prompts_config
        system_prompts = prompts_config.get("system_prompts", {})

        result = {}
        for key, config in system_prompts.items():
            result[key] = {
                "description": config.get("description", ""),
                "variables": config.get("variables", []),
                "max_tokens": config.get("max_tokens"),
                "temperature": config.get("temperature")
            }
        return {
            "total": len(result),
            "prompts": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing prompts: {str(e)}")


@router.get("/prompts/{prompt_key}", response_model=Dict[str, Any])
async def get_prompt(prompt_key: str, admin_key: str = Depends(verify_admin_key)):
    try:
        prompt_manager = get_prompt_manager(mongo_client)
        config = prompt_manager.get_prompt_config(prompt_key)

        if not config:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_key}' not found")

        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prompt: {str(e)}")


@router.get("/prompts/overrides/list", response_model=List[PromptResponse])
async def list_prompt_overrides(
    use_case: Optional[str] = None,
    admin_key: str = Depends(verify_admin_key)
):

    try:
        prompt_manager = get_prompt_manager(mongo_client)
        overrides = prompt_manager.list_overrides(use_case)

        return [
            PromptResponse(
                prompt_key=override.get("prompt_key"),
                prompt_text=override.get("prompt_text"),
                use_case=override.get("use_case"),
                updated_at=override.get("updated_at").isoformat(),
                updated_by=override.get("updated_by", "unknown")
            )
            for override in overrides
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing overrides: {str(e)}")


@router.post("/prompts/override", status_code=201)
async def create_prompt_override(
    override: PromptOverride,
    admin_key: str = Depends(verify_admin_key)
):
    """
    Create or update a prompt override.

    Args:
        override: Prompt override data

    Returns:
        Success message
    """
    try:
        prompt_manager = get_prompt_manager(mongo_client)

        success = prompt_manager.save_override(
            prompt_key=override.prompt_key,
            prompt_text=override.prompt_text,
            use_case=override.use_case,
            updated_by=override.updated_by
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save prompt override")

        return {
            "message": f"Prompt override for '{override.prompt_key}' saved successfully",
            "prompt_key": override.prompt_key,
            "use_case": override.use_case
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving override: {str(e)}")


@router.delete("/prompts/override/{prompt_key}")
async def delete_prompt_override(
    prompt_key: str,
    use_case: Optional[str] = None,
    admin_key: str = Depends(verify_admin_key)
):

    try:
        prompt_manager = get_prompt_manager(mongo_client)

        success = prompt_manager.delete_override(prompt_key, use_case)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No override found for prompt '{prompt_key}'"
            )

        return {
            "message": f"Prompt override for '{prompt_key}' deleted successfully",
            "prompt_key": prompt_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting override: {str(e)}")


@router.post("/prompts/cache/clear")
async def clear_prompt_cache(admin_key: str = Depends(verify_admin_key)):
    """
    Clear the prompt cache to force reload from database.

    Returns:
        Success message
    """
    try:
        prompt_manager = get_prompt_manager(mongo_client)
        prompt_manager.clear_cache()

        return {"message": "Prompt cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@router.get("/tabs", response_model=List[TabConfigResponse])
async def list_tabs(admin_key: str = Depends(verify_admin_key)):
    try:
        tabs_config = settings.tabs_config

        result = []
        for tab_name, config in tabs_config.items():
            result.append(TabConfigResponse(
                tab_name=tab_name,
                display_name=config.get("display_name", tab_name),
                description=config.get("description", ""),
                enabled=config.get("enabled", True),
                metadata_filters=config.get("metadata_filters", []),
                priority=config.get("priority", 999)
            ))

        # Sort by priority
        result.sort(key=lambda x: x.priority)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tabs: {str(e)}")


@router.get("/tabs/enabled", response_model=List[str])
async def list_enabled_tabs(admin_key: str = Depends(verify_admin_key)):
    try:
        return settings.get_enabled_tabs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing enabled tabs: {str(e)}")


@router.get("/tabs/{tab_name}", response_model=TabConfigResponse)
async def get_tab_config(tab_name: str, admin_key: str = Depends(verify_admin_key)):
    try:
        config = settings.get_tab_config(tab_name)

        if not config:
            raise HTTPException(status_code=404, detail=f"Tab '{tab_name}' not found")

        return TabConfigResponse(
            tab_name=tab_name,
            display_name=config.get("display_name", tab_name),
            description=config.get("description", ""),
            enabled=config.get("enabled", True),
            metadata_filters=config.get("metadata_filters", []),
            priority=config.get("priority", 999)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tab config: {str(e)}")



@router.get("/metadata/config", response_model=Dict[str, Any])
async def get_metadata_config(admin_key: str = Depends(verify_admin_key)):
    try:
        return settings.metadata_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata config: {str(e)}")


@router.get("/metadata/filters/important", response_model=List[str])
async def get_important_filters(admin_key: str = Depends(verify_admin_key)):
    try:
        metadata_config = settings.metadata_config
        return metadata_config.get("important_keys", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving important filters: {str(e)}")

@router.get("/llm/providers", response_model=List[str])
async def list_llm_providers(admin_key: str = Depends(verify_admin_key)):
    try:
        return LLMFactory.get_available_providers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing providers: {str(e)}")


@router.get("/llm/providers/{provider}/models", response_model=List[str])
async def list_provider_models(provider: str, admin_key: str = Depends(verify_admin_key)):
    try:
        return LLMFactory.get_provider_models(provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.get("/llm/use-cases", response_model=List[str])
async def list_use_cases(admin_key: str = Depends(verify_admin_key)):
    try:
        return LLMFactory.get_use_cases()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing use cases: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(admin_key: str = Depends(verify_admin_key)):
    components = {}
    warnings = []

    try:
        _ = settings.application_config
        components["application_config"] = "OK"
    except Exception as e:
        components["application_config"] = f"ERROR: {e}"
        warnings.append(f"Application config error: {e}")

    try:
        _ = settings.prompts_config
        components["prompts_config"] = "OK"
    except Exception as e:
        components["prompts_config"] = f"ERROR: {e}"
        warnings.append(f"Prompts config error: {e}")

    try:
        _ = settings.llm_providers_config
        components["llm_providers_config"] = "OK"
    except Exception as e:
        components["llm_providers_config"] = f"ERROR: {e}"
        warnings.append(f"LLM providers config error: {e}")

    # Check database
    try:
        mongo_client.list_database_names()
        components["database"] = "OK"
    except Exception as e:
        components["database"] = f"ERROR: {e}"
        warnings.append(f"Database connection error: {e}")

    # Check vector stores
    try:
        vector_stores = get_vector_stores_v2()
        components["vector_stores"] = f"OK ({len(vector_stores)} stores)"
    except Exception as e:
        components["vector_stores"] = f"ERROR: {e}"
        warnings.append(f"Vector stores error: {e}")

    # Check LLM providers
    try:
        providers = LLMFactory.get_available_providers()
        components["llm_providers"] = f"OK ({len(providers)} enabled)"
    except Exception as e:
        components["llm_providers"] = f"ERROR: {e}"
        warnings.append(f"LLM providers error: {e}")

    # Determine overall status
    has_errors = any("ERROR" in status for status in components.values())
    status = "DEGRADED" if warnings and not has_errors else ("ERROR" if has_errors else "OK")

    return HealthCheckResponse(
        status=status,
        components=components,
        configuration={
            "enabled_tabs": settings.get_enabled_tabs(),
            "vector_store_backend": settings.vector_store,
            "default_llm_provider": settings.llm_providers_config.get("default_provider"),
            "auto_detect_metadata": settings.metadata_config.get("auto_detect")
        },
        warnings=warnings
    )


@router.get("/config/reload")
async def reload_configuration(admin_key: str = Depends(verify_admin_key)):
    """
    Reload configuration from JSON files.

    Note: This clears LRU caches to force reload.

    Returns:
        Success message
    """
    try:
        # Clear configuration caches
        settings._load_json_config.cache_clear()

        # Clear prompt cache
        prompt_manager = get_prompt_manager(mongo_client)
        prompt_manager.clear_cache()

        # Clear tab cache
        from app.services.tab_manager import get_tab_manager
        tab_manager = get_tab_manager(mongo_client)
        tab_manager.clear_cache()

        return {
            "message": "Configuration reloaded successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading configuration: {str(e)}")


class TabCreateRequest(BaseModel):
    tab_name: str = Field(..., description="Unique tab identifier (lowercase, hyphenated)")
    display_name: str = Field(..., description="Human-readable tab name")
    description: str = Field(..., description="Tab description")
    enabled: bool = Field(default=True, description="Whether tab is enabled")
    priority: int = Field(default=999, description="Tab display priority (lower = higher priority)")
    metadata_filters: List[str] = Field(default=[], description="Metadata filter keys for this tab")
    updated_by: str = Field(default="admin", description="User creating the tab")


class TabUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, description="Human-readable tab name")
    description: Optional[str] = Field(None, description="Tab description")
    enabled: Optional[bool] = Field(None, description="Whether tab is enabled")
    priority: Optional[int] = Field(None, description="Tab display priority")
    metadata_filters: Optional[List[str]] = Field(None, description="Metadata filter keys")
    updated_by: str = Field(default="admin", description="User updating the tab")


@router.post("/tabs/create", status_code=201)
async def create_tab(
    request: TabCreateRequest,
    admin_key: str = Depends(verify_admin_key)
):
    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)

        # Build config
        config = {
            "display_name": request.display_name,
            "description": request.description,
            "enabled": request.enabled,
            "priority": request.priority,
            "metadata_filters": request.metadata_filters
        }

        # Save override
        success = tab_manager.save_tab_override(
            tab_name=request.tab_name,
            config=config,
            updated_by=request.updated_by
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create tab")

        return {
            "message": f"Tab '{request.tab_name}' created successfully",
            "tab_name": request.tab_name,
            "config": config
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating tab: {str(e)}")


@router.put("/tabs/{tab_name}/update")
async def update_tab(
    tab_name: str,
    request: TabUpdateRequest,
    admin_key: str = Depends(verify_admin_key)
):

    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)

        # Get current config
        current_config = tab_manager.get_tab_config(tab_name)
        if not current_config:
            raise HTTPException(status_code=404, detail=f"Tab '{tab_name}' not found")

        # Update only provided fields
        if request.display_name is not None:
            current_config["display_name"] = request.display_name
        if request.description is not None:
            current_config["description"] = request.description
        if request.enabled is not None:
            current_config["enabled"] = request.enabled
        if request.priority is not None:
            current_config["priority"] = request.priority
        if request.metadata_filters is not None:
            current_config["metadata_filters"] = request.metadata_filters

        # Save override
        success = tab_manager.save_tab_override(
            tab_name=tab_name,
            config=current_config,
            updated_by=request.updated_by
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update tab")

        return {
            "message": f"Tab '{tab_name}' updated successfully",
            "tab_name": tab_name,
            "config": current_config
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating tab: {str(e)}")


@router.patch("/tabs/{tab_name}/toggle")
async def toggle_tab(
    tab_name: str,
    enabled: bool,
    admin_key: str = Depends(verify_admin_key)
):
    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)

        # Toggle tab
        success = tab_manager.toggle_tab(tab_name, enabled, updated_by="admin")

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Tab '{tab_name}' not found or toggle failed"
            )

        status = "enabled" if enabled else "disabled"
        return {
            "message": f"Tab '{tab_name}' {status} successfully",
            "tab_name": tab_name,
            "enabled": enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error toggling tab: {str(e)}")


@router.delete("/tabs/{tab_name}/override")
async def delete_tab_override(
    tab_name: str,
    admin_key: str = Depends(verify_admin_key)
):
    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)

        success = tab_manager.delete_tab_override(tab_name)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No override found for tab '{tab_name}'"
            )

        return {
            "message": f"Override for tab '{tab_name}' deleted successfully (reverted to default)",
            "tab_name": tab_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting tab override: {str(e)}")



@router.get("/tabs/validate")
async def validate_tabs(admin_key: str = Depends(verify_admin_key)):
    """
    Validate all tab configurations.

    Returns:
        Validation results
    """
    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)
        validation = tab_manager.validate_all_tabs()

        return {
            "valid_count": len(validation["valid"]),
            "invalid_count": len(validation["invalid"]),
            "valid_tabs": validation["valid"],
            "invalid_tabs": validation["invalid"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating tabs: {str(e)}")


@router.post("/tabs/cache/clear")
async def clear_tab_cache(admin_key: str = Depends(verify_admin_key)):
    """
    Clear tab configuration cache.

    Returns:
        Success message
    """
    try:
        from app.services.tab_manager import get_tab_manager

        tab_manager = get_tab_manager(mongo_client)
        tab_manager.clear_cache()

        return {"message": "Tab cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing tab cache: {str(e)}")


@router.get("/query-cache/stats")
async def get_query_cache_stats(admin_key: str = Depends(verify_admin_key)):
    """
    Get query cache statistics.
    
    Shows cache hit rate, size, and estimated API cost savings.
    
    Returns:
        Cache statistics
    """
    try:
        from app.services.query_cache import get_query_cache
        
        query_cache = get_query_cache()
        stats = query_cache.get_stats()
        
        return {
            "cache_statistics": stats,
            "configuration": {
                "similarity_threshold": query_cache.similarity_threshold,
                "ttl_seconds": query_cache.ttl,
                "namespace": query_cache.namespace
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting query cache stats: {str(e)}")


@router.post("/query-cache/clear")
async def clear_query_cache(admin_key: str = Depends(verify_admin_key)):
    """
    Clear query cache.
    
    Removes all cached query embeddings and results.
    
    Returns:
        Success message with previous stats
    """
    try:
        from app.services.query_cache import get_query_cache
        
        query_cache = get_query_cache()
        previous_stats = query_cache.get_stats()
        query_cache.clear()
        
        return {
            "message": "Query cache cleared successfully",
            "previous_stats": previous_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing query cache: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing tab cache: {str(e)}")
