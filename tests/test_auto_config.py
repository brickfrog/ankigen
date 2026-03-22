import pytest
from unittest.mock import AsyncMock
from pydantic import ValidationError

from ankigen.auto_config import AutoConfigService
from ankigen.agents.schemas import AutoConfigSchema


@pytest.mark.anyio
async def test_auto_configure_empty_subject():
    """Test that auto_configure returns an empty dict for an empty subject."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    result = await service.auto_configure("", openai_client)
    assert result == {}


@pytest.mark.anyio
async def test_auto_configure_whitespace_subject():
    """Test that auto_configure returns an empty dict for a whitespace subject."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    result = await service.auto_configure("   ", openai_client)
    assert result == {}


@pytest.mark.anyio
async def test_auto_configure_success_with_library(mocker):
    """Test successful auto-configuration with a detected library."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    # Mock analyze_subject
    mock_config = AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus="dataframe",
        topic_number=5,
        topics_list=["t1", "t2", "t3", "t4", "t5"],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="api",
        scope="medium",
        rationale="rationale",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # Mock context7_client.resolve_library_id
    mocker.patch.object(
        service.context7_client, "resolve_library_id", return_value="/pandas/pandas"
    )

    result = await service.auto_configure("pandas basics", openai_client)

    assert result["library_name"] == "pandas"
    assert result["library_topic"] == "dataframe"
    assert result["topic_number"] == 5
    assert result["analysis_metadata"]["library_found"] is True
    assert result["analysis_metadata"]["context7_id"] == "/pandas/pandas"


@pytest.mark.anyio
async def test_auto_configure_success_no_library(mocker):
    """Test successful auto-configuration when no library is detected."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    # Mock analyze_subject with empty library_search_term
    mock_config = AutoConfigSchema(
        library_search_term="",
        documentation_focus=None,
        topic_number=3,
        topics_list=["t1", "t2", "t3"],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="narrow",
        rationale="rationale",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # Mock resolve_library_id (should not be called if library_search_term is empty,
    # but good to mock just in case)
    mock_resolve = mocker.patch.object(service.context7_client, "resolve_library_id")

    result = await service.auto_configure("philosophy", openai_client)

    assert result["library_name"] == ""
    assert result["topic_number"] == 3
    assert result["analysis_metadata"]["library_found"] is False
    mock_resolve.assert_not_called()


@pytest.mark.anyio
async def test_auto_configure_library_not_found(mocker):
    """Test auto-configuration when a library is detected but not found in Context7."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="nonexistent-lib",
        documentation_focus=None,
        topic_number=3,
        topics_list=["t1", "t2", "t3"],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="api",
        scope="narrow",
        rationale="rationale",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # resolve_library_id returns None
    mocker.patch.object(
        service.context7_client, "resolve_library_id", return_value=None
    )

    result = await service.auto_configure("nonexistent-lib basics", openai_client)

    assert result["library_name"] == ""
    assert result["analysis_metadata"]["library_found"] is False
    assert result["analysis_metadata"]["context7_id"] is None


@pytest.mark.anyio
async def test_auto_configure_context7_error(mocker):
    """Test auto-configuration when Context7 client raises an exception."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus=None,
        topic_number=3,
        topics_list=["t1", "t2", "t3"],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="api",
        scope="narrow",
        rationale="rationale",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # resolve_library_id raises exception
    mocker.patch.object(
        service.context7_client,
        "resolve_library_id",
        side_effect=Exception("Context7 down"),
    )

    result = await service.auto_configure("pandas basics", openai_client)

    # Should handle gracefully
    assert result["library_name"] == ""
    assert result["analysis_metadata"]["library_found"] is False


@pytest.mark.anyio
async def test_analyze_subject_success(mocker):
    """Test successful subject analysis using the LLM."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="react",
        documentation_focus="hooks",
        topic_number=4,
        topics_list=["useState", "useEffect", "useContext", "useReducer"],
        cards_per_topic=10,
        learning_preferences="focus on hooks",
        generate_cloze=True,
        model_choice="gpt-5.2-thinking",
        subject_type="syntax",
        scope="medium",
        rationale="React hooks are important",
    )

    mock_call = mocker.patch(
        "ankigen.auto_config.structured_agent_call", return_value=mock_config
    )

    result = await service.analyze_subject("React hooks tutorial", openai_client)

    assert result == mock_config
    assert mock_call.call_args[1]["output_type"] == AutoConfigSchema


@pytest.mark.anyio
async def test_analyze_subject_with_topic_count_override(mocker):
    """Test subject analysis with a target topic count override."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="",
        documentation_focus=None,
        topic_number=10,
        topics_list=[f"Topic {i}" for i in range(10)],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="broad",
        rationale="rationale",
    )

    mock_call = mocker.patch(
        "ankigen.auto_config.structured_agent_call", return_value=mock_config
    )

    await service.analyze_subject(
        "Computer Science", openai_client, target_topic_count=10
    )

    # Verify that the topic_count_instruction was likely included in the system prompt
    system_prompt = mock_call.call_args[1]["instructions"]
    assert "exactly 10 topics" in system_prompt


@pytest.mark.anyio
async def test_analyze_subject_fallback_on_error(mocker):
    """Test that analyze_subject returns fallback values on LLM error."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mocker.patch(
        "ankigen.auto_config.structured_agent_call", side_effect=Exception("LLM failed")
    )

    result = await service.analyze_subject("Broken Subject", openai_client)

    assert result.topic_number == 6
    assert len(result.topics_list) == 6
    assert "Broken Subject" in result.topics_list[0]
    assert result.rationale == "Using default settings due to analysis error"


def test_auto_config_schema_valid():
    """Test that AutoConfigSchema validates correct data."""
    data = {
        "library_search_term": "numpy",
        "documentation_focus": "arrays",
        "topic_number": 5,
        "topics_list": ["intro", "creation", "indexing", "slicing", "broadcasting"],
        "cards_per_topic": 8,
        "learning_preferences": "focus on performance",
        "generate_cloze": True,
        "model_choice": "gpt-5.2-auto",
        "subject_type": "api",
        "scope": "medium",
        "rationale": "Numerical computing with numpy",
    }
    schema = AutoConfigSchema(**data)
    assert schema.library_search_term == "numpy"


def test_auto_config_schema_invalid_topic_number():
    """Test that AutoConfigSchema rejects invalid topic numbers."""
    data = {
        "library_search_term": "numpy",
        "documentation_focus": "arrays",
        "topic_number": 1,  # Too low, min is 2
        "topics_list": ["only one"],
        "cards_per_topic": 8,
        "learning_preferences": "focus on performance",
        "generate_cloze": True,
        "model_choice": "gpt-5.2-auto",
        "subject_type": "api",
        "scope": "medium",
        "rationale": "Numerical computing with numpy",
    }
    with pytest.raises(ValidationError):
        AutoConfigSchema(**data)


@pytest.mark.anyio
async def test_auto_configure_metadata_passing(mocker):
    """Verify that all relevant metadata is correctly passed to the UI config."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus="plotting",
        topic_number=4,
        topics_list=["line", "bar", "hist", "scatter"],
        cards_per_topic=7,
        learning_preferences="visualization",
        generate_cloze=False,
        model_choice="gpt-5.2-instant",
        subject_type="practical",
        scope="medium",
        rationale="Pandas plotting is useful",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)
    mocker.patch.object(
        service.context7_client, "resolve_library_id", return_value="pandas_id"
    )

    result = await service.auto_configure("pandas plotting", openai_client)

    assert result["library_topic"] == "plotting"
    assert result["cards_per_topic"] == 7
    assert result["preference_prompt"] == "visualization"
    assert result["generate_cloze_checkbox"] is False
    assert result["model_choice"] == "gpt-5.2-instant"
    assert result["analysis_metadata"]["subject_type"] == "practical"
    assert result["analysis_metadata"]["scope"] == "medium"
    assert result["analysis_metadata"]["rationale"] == "Pandas plotting is useful"


@pytest.mark.anyio
async def test_auto_configure_no_doc_focus(mocker):
    """Test auto-configuration when documentation_focus is None."""
    service = AutoConfigService()
    openai_client = AsyncMock()

    mock_config = AutoConfigSchema(
        library_search_term="flask",
        documentation_focus=None,
        topic_number=3,
        topics_list=["t1", "t2", "t3"],
        cards_per_topic=5,
        learning_preferences="pref",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="api",
        scope="narrow",
        rationale="rationale",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)
    mocker.patch.object(
        service.context7_client, "resolve_library_id", return_value="flask_id"
    )

    result = await service.auto_configure("flask", openai_client)

    assert result["library_topic"] == ""
