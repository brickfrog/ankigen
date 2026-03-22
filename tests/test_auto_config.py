import pytest
from unittest.mock import AsyncMock, MagicMock
from openai import AsyncOpenAI
from ankigen.auto_config import AutoConfigService
from ankigen.agents.schemas import AutoConfigSchema

# --- AutoConfigService Tests ---


@pytest.mark.anyio
async def test_analyze_subject_success(mocker):
    service = AutoConfigService()
    mock_call = mocker.patch(
        "ankigen.auto_config.structured_agent_call", new_callable=AsyncMock
    )

    mock_config = AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus="dataframe basics",
        topic_number=3,
        topics_list=["t1", "t2", "t3"],
        cards_per_topic=5,
        learning_preferences="prefs",
        generate_cloze=True,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="medium",
        rationale="rat",
    )
    mock_call.return_value = mock_config

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.analyze_subject("pandas basics", mock_client)

    assert result.library_search_term == "pandas"
    assert result.topic_number == 3
    assert len(result.topics_list) == 3
    mock_call.assert_called_once()


@pytest.mark.anyio
async def test_analyze_subject_with_target_count(mocker):
    service = AutoConfigService()
    mock_call = mocker.patch(
        "ankigen.auto_config.structured_agent_call", new_callable=AsyncMock
    )
    mock_call.return_value = MagicMock(spec=AutoConfigSchema)

    mock_client = MagicMock(spec=AsyncOpenAI)
    await service.analyze_subject("pandas basics", mock_client, target_topic_count=5)

    # Verify that instructions in system_prompt includes the target count
    args, kwargs = mock_call.call_args
    assert "You MUST set topic_number to 5" in kwargs["instructions"]


@pytest.mark.anyio
async def test_analyze_subject_error_fallback(mocker):
    service = AutoConfigService()
    mocker.patch(
        "ankigen.auto_config.structured_agent_call", side_effect=Exception("API Error")
    )

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.analyze_subject("error subject", mock_client)

    assert result.topic_number == 6
    assert result.rationale == "Using default settings due to analysis error"
    assert "error subject" in result.topics_list[0]


@pytest.mark.anyio
async def test_auto_configure_empty_subject():
    service = AutoConfigService()
    mock_client = MagicMock(spec=AsyncOpenAI)

    assert await service.auto_configure("", mock_client) == {}
    assert await service.auto_configure("   ", mock_client) == {}


@pytest.mark.anyio
async def test_auto_configure_success(mocker):
    service = AutoConfigService()

    # Mock analyze_subject
    mock_config = AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus="df",
        topic_number=2,
        topics_list=["t1", "t2"],
        cards_per_topic=10,
        learning_preferences="prefs",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="medium",
        rationale="rat",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # Mock context7_client.resolve_library_id
    service.context7_client.resolve_library_id = AsyncMock(
        return_value="/python/pandas"
    )

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.auto_configure("pandas", mock_client)

    assert result["library_name"] == "pandas"
    assert result["topic_number"] == 2
    assert result["analysis_metadata"]["context7_id"] == "/python/pandas"
    assert result["analysis_metadata"]["library_found"] is True


@pytest.mark.anyio
async def test_auto_configure_library_not_found(mocker):
    service = AutoConfigService()

    mock_config = AutoConfigSchema(
        library_search_term="unknown_lib",
        documentation_focus="df",
        topic_number=2,
        topics_list=["t1", "t2"],
        cards_per_topic=10,
        learning_preferences="prefs",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="medium",
        rationale="rat",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)

    # Mock context7_client.resolve_library_id to return None
    service.context7_client.resolve_library_id = AsyncMock(return_value=None)

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.auto_configure("unknown", mock_client)

    assert result["library_name"] == ""
    assert result["analysis_metadata"]["library_found"] is False


@pytest.mark.anyio
async def test_auto_configure_context7_error(mocker):
    service = AutoConfigService()

    mocker.patch.object(
        service, "analyze_subject", return_value=MagicMock(library_search_term="pandas")
    )
    service.context7_client.resolve_library_id = AsyncMock(
        side_effect=Exception("C7 Error")
    )

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.auto_configure("pandas", mock_client)

    assert result["analysis_metadata"]["library_found"] is False
    assert result["analysis_metadata"]["context7_id"] is None


def test_service_init():
    service = AutoConfigService()
    assert service.context7_client is not None


@pytest.mark.anyio
async def test_analyze_subject_call_args(mocker):
    service = AutoConfigService()
    mock_call = mocker.patch(
        "ankigen.auto_config.structured_agent_call", new_callable=AsyncMock
    )
    mock_call.return_value = MagicMock(spec=AutoConfigSchema)

    mock_client = MagicMock(spec=AsyncOpenAI)
    await service.analyze_subject("testing", mock_client)

    args, kwargs = mock_call.call_args
    assert kwargs["model"] == "gpt-5.2"
    assert kwargs["output_type"] == AutoConfigSchema
    assert kwargs["temperature"] == 0.3


@pytest.mark.anyio
async def test_auto_configure_no_library_detected(mocker):
    service = AutoConfigService()

    mock_config = AutoConfigSchema(
        library_search_term="",  # No library
        documentation_focus=None,
        topic_number=2,
        topics_list=["t1", "t2"],
        cards_per_topic=5,
        learning_preferences="prefs",
        generate_cloze=False,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="medium",
        rationale="rat",
    )
    mocker.patch.object(service, "analyze_subject", return_value=mock_config)
    service.context7_client.resolve_library_id = AsyncMock()

    mock_client = MagicMock(spec=AsyncOpenAI)
    result = await service.auto_configure("general topic", mock_client)

    assert result["library_name"] == ""
    service.context7_client.resolve_library_id.assert_not_called()
