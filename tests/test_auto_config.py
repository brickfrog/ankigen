import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ankigen.auto_config import AutoConfigService
from ankigen.agents.schemas import AutoConfigSchema


@pytest.fixture
def auto_config_service():
    return AutoConfigService()


@pytest.fixture
def mock_config_response():
    return AutoConfigSchema(
        library_search_term="pandas",
        documentation_focus="dataframe basics",
        topic_number=5,
        topics_list=["t1", "t2", "t3", "t4", "t5"],
        cards_per_topic=10,
        learning_preferences="Focus on basics",
        generate_cloze=True,
        model_choice="gpt-5.2-auto",
        subject_type="concepts",
        scope="medium",
        rationale="test rationale",
    )


@pytest.mark.anyio
async def test_analyze_subject_success(auto_config_service, mock_config_response):
    mock_client = MagicMock()
    with patch(
        "ankigen.auto_config.structured_agent_call",
        AsyncMock(return_value=mock_config_response),
    ):
        config = await auto_config_service.analyze_subject("pandas basics", mock_client)

        assert config.library_search_term == "pandas"
        assert config.topic_number == 5
        assert len(config.topics_list) == 5


@pytest.mark.anyio
async def test_analyze_subject_error_fallback(auto_config_service):
    mock_client = MagicMock()
    with patch(
        "ankigen.auto_config.structured_agent_call",
        AsyncMock(side_effect=Exception("API Error")),
    ):
        config = await auto_config_service.analyze_subject("any subject", mock_client)

        # Check fallback values
        assert config.topic_number == 6
        assert len(config.topics_list) == 6
        assert "analysis error" in config.rationale


@pytest.mark.anyio
async def test_analyze_subject_with_target_count(
    auto_config_service, mock_config_response
):
    mock_client = MagicMock()
    with patch(
        "ankigen.auto_config.structured_agent_call",
        AsyncMock(return_value=mock_config_response),
    ) as mock_call:
        await auto_config_service.analyze_subject(
            "subject", mock_client, target_topic_count=3
        )

        # Verify instructions passed to agent
        args, kwargs = mock_call.call_args
        assert "exactly 3 topics" in kwargs["instructions"]


@pytest.mark.anyio
async def test_auto_configure_success(auto_config_service, mock_config_response):
    mock_client = MagicMock()
    auto_config_service.context7_client.resolve_library_id = AsyncMock(
        return_value="/pandas/docs"
    )

    with patch.object(
        auto_config_service,
        "analyze_subject",
        AsyncMock(return_value=mock_config_response),
    ):
        ui_config = await auto_config_service.auto_configure(
            "pandas basics", mock_client
        )

        assert ui_config["library_name"] == "pandas"
        assert ui_config["analysis_metadata"]["context7_id"] == "/pandas/docs"
        assert ui_config["topic_number"] == 5


@pytest.mark.anyio
async def test_auto_configure_empty_subject(auto_config_service):
    mock_client = MagicMock()
    result = await auto_config_service.auto_configure("", mock_client)
    assert result == {}


@pytest.mark.anyio
async def test_auto_configure_no_library(auto_config_service, mock_config_response):
    mock_client = MagicMock()
    mock_config_response.library_search_term = ""

    with patch.object(
        auto_config_service,
        "analyze_subject",
        AsyncMock(return_value=mock_config_response),
    ):
        ui_config = await auto_config_service.auto_configure(
            "general subject", mock_client
        )

        assert ui_config["library_name"] == ""
        assert ui_config["analysis_metadata"]["library_found"] is False


@pytest.mark.anyio
async def test_auto_configure_context7_failure(
    auto_config_service, mock_config_response
):
    mock_client = MagicMock()
    auto_config_service.context7_client.resolve_library_id = AsyncMock(
        side_effect=Exception("Context7 Down")
    )

    with patch.object(
        auto_config_service,
        "analyze_subject",
        AsyncMock(return_value=mock_config_response),
    ):
        ui_config = await auto_config_service.auto_configure("pandas", mock_client)

        assert ui_config["analysis_metadata"]["library_found"] is False
        assert ui_config["library_name"] == ""


@pytest.mark.anyio
async def test_auto_configure_no_library_id(auto_config_service, mock_config_response):
    mock_client = MagicMock()
    auto_config_service.context7_client.resolve_library_id = AsyncMock(
        return_value=None
    )

    with patch.object(
        auto_config_service,
        "analyze_subject",
        AsyncMock(return_value=mock_config_response),
    ):
        ui_config = await auto_config_service.auto_configure("unknown lib", mock_client)

        assert ui_config["analysis_metadata"]["library_found"] is False
        assert ui_config["library_name"] == ""


@pytest.mark.anyio
async def test_auto_config_ui_config_structure(
    auto_config_service, mock_config_response
):
    mock_client = MagicMock()
    auto_config_service.context7_client.resolve_library_id = AsyncMock(
        return_value="/lib/id"
    )

    with patch.object(
        auto_config_service,
        "analyze_subject",
        AsyncMock(return_value=mock_config_response),
    ):
        ui_config = await auto_config_service.auto_configure("subject", mock_client)

        expected_keys = {
            "library_name",
            "library_topic",
            "topic_number",
            "topics_list",
            "cards_per_topic",
            "preference_prompt",
            "generate_cloze_checkbox",
            "model_choice",
            "analysis_metadata",
        }
        assert set(ui_config.keys()) == expected_keys

        metadata_keys = {
            "subject_type",
            "scope",
            "rationale",
            "library_found",
            "context7_id",
        }
        assert set(ui_config["analysis_metadata"].keys()) == metadata_keys


@pytest.mark.anyio
async def test_auto_config_service_init(auto_config_service):
    assert auto_config_service.context7_client is not None
