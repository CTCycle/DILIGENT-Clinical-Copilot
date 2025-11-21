from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import Any

from langchain_community.agent_toolkits.nla import toolkit
from nicegui import ui

from DILIGENT.src.app.client.services import (
    DILISessionEndpointService,
    ModelPullEndpointService,
    RuntimeSettings,
    SettingsService,
)
from DILIGENT.src.app.client.layouts import (
    CARD_BASE_CLASSES,
    INTERFACE_THEME_CSS,
    JSON_CARD_CLASSES,
    PAGE_CONTAINER_CLASSES,
    VISIT_DATE_CSS,
    VISIT_DATE_ELEMENT_ID,
    VISIT_DATE_LOCALE_JS,
)
from DILIGENT.src.packages.configurations import configurations
from DILIGENT.src.packages.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    PARSING_MODEL_CHOICES,
)

export_attribute = "export_path"

CLOUD_PROVIDERS: list[str] = [key for key in CLOUD_MODEL_CHOICES]
ui_settings = configurations.client.ui


# [TOOLKIT]
###############################################################################
class InterfaceToolkit:    

    # -------------------------------------------------------------------------
    def update_json_display(self, container: Any, code: Any, payload: Any) -> None:
        if payload is not None:
            container.visible = True
            formatted = json.dumps(payload, ensure_ascii=False, indent=2)
            code.set_content(formatted)
        else:
            container.visible = False
            code.set_content("")

    # -------------------------------------------------------------------------
    def update_cloud_controls(
        self,
        enabled: bool,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        parsing_model_dropdown: Any,
        clinical_model_dropdown: Any,
        temperature_input: Any,
        reasoning_checkbox: Any,
        pull_models_button: Any | None,
    ) -> None:
        if enabled:
            llm_provider_dropdown.enable()
            cloud_model_dropdown.enable()
            if pull_models_button is not None:
                pull_models_button.disable()
            parsing_model_dropdown.disable()
            clinical_model_dropdown.disable()
            temperature_input.disable()
            reasoning_checkbox.disable()
        else:
            llm_provider_dropdown.disable()
            cloud_model_dropdown.disable()
            if pull_models_button is not None:
                pull_models_button.enable()
            parsing_model_dropdown.enable()
            clinical_model_dropdown.enable()
            temperature_input.enable()
            reasoning_checkbox.enable()

    # -------------------------------------------------------------------------
    def set_session_progress_state(
        self,
        markdown_output: Any,
        session_spinner: Any,
        loading: bool,
    ) -> None:
        session_spinner.visible = loading
        session_spinner.update()
        markdown_output.visible = not loading
        markdown_output.update()


# [INTERFACE CONTROLLER]
###############################################################################
class InterfaceService:
    def __init__(
        self,
        settings_controller: SettingsService,
        model_pull_controller: ModelPullEndpointService,
        dili_session_controller: DILISessionEndpointService,
        toolkit: InterfaceToolkit,
    ) -> None:
        self.settings_controller = settings_controller
        self.model_pull_controller = model_pull_controller
        self.dili_session_controller = dili_session_controller
        self.toolkit = toolkit    

    # -------------------------------------------------------------------------
    async def handle_toggle_cloud_services(
        self,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        parsing_model_dropdown: Any,
        clinical_model_dropdown: Any,
        temperature_input: Any,
        reasoning_checkbox: Any,
        pull_models_button: Any,
        event: Any,
    ) -> None:
        enabled = bool(event.value)
        selection = self.settings_controller.resolve_cloud_selection(
            str(llm_provider_dropdown.value or ""),
            str(cloud_model_dropdown.value or ""),
        )
        llm_provider_dropdown.value = selection["provider"]
        llm_provider_dropdown.update()
        cloud_model_dropdown.set_options(selection["models"])
        cloud_model_dropdown.value = selection["model"]
        cloud_model_dropdown.update()
        self.toolkit.update_cloud_controls(
            enabled,
            llm_provider_dropdown,
            cloud_model_dropdown,
            parsing_model_dropdown,
            clinical_model_dropdown,
            temperature_input,
            reasoning_checkbox,
            pull_models_button,
        )

    # -------------------------------------------------------------------------
    async def handle_cloud_provider_change(
        self,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        event: Any,
    ) -> None:
        selection = self.settings_controller.resolve_cloud_selection(
            str(event.value or ""),
            str(cloud_model_dropdown.value or ""),
        )
        llm_provider_dropdown.value = selection["provider"]
        llm_provider_dropdown.update()
        cloud_model_dropdown.set_options(selection["models"])
        cloud_model_dropdown.value = selection["model"]
        cloud_model_dropdown.update()

    # -------------------------------------------------------------------------
    async def handle_visit_date_change(
        self,
        visit_date_input: Any,
        event: Any,
    ) -> None:
        normalized = self.settings_controller.normalize_visit_date_component(
            event.value
        )
        visit_date_input.value = (
            "" if normalized is None else normalized.date().isoformat()
        )
        visit_date_input.update()

    # -------------------------------------------------------------------------
    async def handle_pull_models_click(
        self,
        use_cloud_services_checkbox: Any,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        parsing_model_dropdown: Any,
        clinical_model_dropdown: Any,
        temperature_input: Any,
        reasoning_checkbox: Any,
        markdown_output: Any,
        json_container: Any,
        json_code: Any,
        event: Any,
    ) -> None:
        settings = RuntimeSettings(
            use_cloud_services=bool(use_cloud_services_checkbox.value),
            provider=str(llm_provider_dropdown.value or ""),
            cloud_model=str(cloud_model_dropdown.value or ""),
            parsing_model=str(parsing_model_dropdown.value or ""),
            clinical_model=str(clinical_model_dropdown.value or ""),
            temperature=temperature_input.value,
            reasoning=bool(reasoning_checkbox.value),
        )
        result = await self.model_pull_controller.pull_selected_models(settings)
        markdown_output.set_content(result.get("message") or "")
        self.toolkit.update_json_display(json_container, json_code, result.get("json"))

    # -------------------------------------------------------------------------
    async def handle_run_workflow(
        self,
        patient_name_input: Any,
        visit_date_input: Any,
        anamnesis_input: Any,
        drugs_input: Any,
        alt_input: Any,
        alt_max_input: Any,
        alp_input: Any,
        alp_max_input: Any,
        use_rag_checkbox: Any,
        use_cloud_services_checkbox: Any,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        parsing_model_dropdown: Any,
        clinical_model_dropdown: Any,
        temperature_input: Any,
        reasoning_checkbox: Any,
        markdown_output: Any,
        session_spinner: Any,
        json_container: Any,
        json_code: Any,
        export_button: Any,
        event: Any,
    ) -> None:
        settings = RuntimeSettings(
            use_cloud_services=bool(use_cloud_services_checkbox.value),
            provider=str(llm_provider_dropdown.value or ""),
            cloud_model=str(cloud_model_dropdown.value or ""),
            parsing_model=str(parsing_model_dropdown.value or ""),
            clinical_model=str(clinical_model_dropdown.value or ""),
            temperature=temperature_input.value,
            reasoning=bool(reasoning_checkbox.value),
        )
        self.toolkit.set_session_progress_state(markdown_output, session_spinner, True)
        try:
            result = await self.dili_session_controller.run_DILI_session(
                patient_name_input.value,
                visit_date_input.value,
                anamnesis_input.value,
                drugs_input.value,
                alt_input.value,
                alt_max_input.value,
                alp_input.value,
                alp_max_input.value,
                use_rag_checkbox.value,
                settings,
            )
        except Exception:
            self.toolkit.set_session_progress_state(markdown_output, session_spinner, False)
            raise

        markdown_output.set_content(result.get("message") or "")
        self.toolkit.update_json_display(json_container, json_code, result.get("json"))
        export_path = result.get(export_attribute)
        setattr(export_button, export_attribute, export_path)
        export_button.enable() if export_path else export_button.disable()
        self.toolkit.set_session_progress_state(markdown_output, session_spinner, False)

    # -------------------------------------------------------------------------
    async def handle_clear_click(
        self,
        patient_name_input: Any,
        visit_date_input: Any,
        anamnesis_input: Any,
        drugs_input: Any,
        alt_input: Any,
        alt_max_input: Any,
        alp_input: Any,
        alp_max_input: Any,
        use_rag_checkbox: Any,
        use_cloud_services_checkbox: Any,
        llm_provider_dropdown: Any,
        cloud_model_dropdown: Any,
        pull_models_button: Any,
        parsing_model_dropdown: Any,
        clinical_model_dropdown: Any,
        temperature_input: Any,
        reasoning_checkbox: Any,
        markdown_output: Any,
        session_spinner: Any,
        json_container: Any,
        json_code: Any,
        export_button: Any,
        event: Any,
    ) -> None:
        defaults = self.settings_controller.clear_session_fields()
        settings = defaults["settings"]
        patient_name_input.value = defaults["patient_name"]
        patient_name_input.update()
        visit_date_value = defaults["visit_date"]
        visit_date_input.value = (
            visit_date_value.date().isoformat() if visit_date_value else ""
        )
        visit_date_input.update()
        anamnesis_input.value = defaults["anamnesis"]
        anamnesis_input.update()
        drugs_input.value = defaults["drugs"]
        drugs_input.update()
        alt_input.value = defaults["alt"]
        alt_input.update()
        alt_max_input.value = defaults["alt_max"]
        alt_max_input.update()
        alp_input.value = defaults["alp"]
        alp_input.update()
        alp_max_input.value = defaults["alp_max"]
        alp_max_input.update()
        use_rag_checkbox.value = defaults["use_rag"]
        use_rag_checkbox.update()
        use_cloud_services_checkbox.value = settings.use_cloud_services
        use_cloud_services_checkbox.update()
        selection = self.settings_controller.resolve_cloud_selection(
            settings.provider, settings.cloud_model
        )
        llm_provider_dropdown.value = selection["provider"]
        llm_provider_dropdown.update()
        cloud_model_dropdown.set_options(selection["models"])
        cloud_model_dropdown.value = selection["model"]
        cloud_model_dropdown.update()
        parsing_model_dropdown.value = settings.parsing_model
        parsing_model_dropdown.update()
        clinical_model_dropdown.value = settings.clinical_model
        clinical_model_dropdown.update()
        temperature_input.value = settings.temperature
        temperature_input.update()
        reasoning_checkbox.value = settings.reasoning
        reasoning_checkbox.update()
        self.toolkit.update_cloud_controls(
            settings.use_cloud_services,
            llm_provider_dropdown,
            cloud_model_dropdown,
            parsing_model_dropdown,
            clinical_model_dropdown,
            temperature_input,
            reasoning_checkbox,
            pull_models_button,
        )
        markdown_output.set_content(defaults["message"])
        self.toolkit.update_json_display(json_container, json_code, defaults.get("json"))
        self.toolkit.set_session_progress_state(markdown_output, session_spinner, False)
        setattr(export_button, export_attribute, defaults.get(export_attribute))
        export_button.disable()

    # -------------------------------------------------------------------------
    async def handle_download_click(self, export_button: Any, event: Any) -> None:
        export_path = getattr(export_button, export_attribute, None)
        if export_path:
            ui.download(export_path)


# [INTERFACE STRUCTURE]
###############################################################################
class InterfaceStructure:
    def __init__(
        self,
        settings_controller: SettingsService,
        controller: InterfaceService,
        toolkit: InterfaceToolkit,
    ) -> None:
        self.settings_controller = settings_controller
        self.controller = controller
        self.toolkit = toolkit

    # -------------------------------------------------------------------------
    def compose_main_page(self) -> None:
        current_settings = self.settings_controller.get_runtime_settings()
        selection = self.settings_controller.resolve_cloud_selection(
            current_settings.provider, current_settings.cloud_model
        )
        provider = selection["provider"]
        cloud_models = selection["models"]
        selected_cloud_model = selection["model"]
        cloud_enabled = current_settings.use_cloud_services

        ui.page_title("DILIGENT Clinical Copilots")
        ui.add_head_html(f"<style>{VISIT_DATE_CSS}{INTERFACE_THEME_CSS}</style>")

        config_toolbar = ui.left_drawer(value=False, fixed=True).props(
            "width=360 overlay bordered elevated"
        )
        with config_toolbar:
            with ui.column().classes("gap-4 p-4 w-full"):
                ui.label("Model Configurations").classes("diligent-card-title")
                use_rag_checkbox = ui.checkbox(
                    "Use Retrieval Augmented Generation (RAG)",
                    value=False,
                ).classes("pt-2")
                use_cloud_services = ui.checkbox(
                    "Use Cloud Services",
                    value=cloud_enabled,
                ).classes("pt-2")
                with ui.column().classes("w-full gap-3"):
                    ui.label("Cloud Configuration").classes("diligent-subtitle")
                    llm_provider_dropdown = ui.select(
                        CLOUD_PROVIDERS,
                        label="Cloud Service",
                        value=provider,
                    ).classes("w-full")
                    llm_provider_dropdown.disable()
                    llm_provider_dropdown.props("dense")
                    cloud_model_dropdown = ui.select(
                        cloud_models,
                        label="Cloud Model",
                        value=selected_cloud_model,
                    ).classes("w-full")
                    cloud_model_dropdown.disable()
                    cloud_model_dropdown.props("dense")
                with ui.column().classes("w-full gap-3"):
                    ui.label("Local Configuration").classes("diligent-subtitle")
                    parsing_model_dropdown = ui.select(
                        PARSING_MODEL_CHOICES,
                        label="Parsing Model",
                        value=current_settings.parsing_model,
                    ).classes("w-full")
                    parsing_model_dropdown.props("dense")
                    clinical_model_dropdown = ui.select(
                        CLINICAL_MODEL_CHOICES,
                        label="Clinical Model",
                        value=current_settings.clinical_model,
                    ).classes("w-full")
                    clinical_model_dropdown.props("dense")
                with ui.column().classes("w-full gap-3"):
                    temperature_input = ui.number(
                        label="Temperature (Ollama)",
                        value=current_settings.temperature,
                        format="%.2f",
                    ).classes("w-full")
                    temperature_input.props("dense")
                    reasoning_checkbox = ui.checkbox(
                        "Enable SDL/Reasoning (Ollama)",
                        value=current_settings.reasoning,
                    ).classes("pt-2")
                pull_models_button = ui.button("Pull Selected Models").classes(
                    "w-full"
                )
            ui.button(
                icon="chevron_left",
                on_click=lambda _: config_toolbar.set_value(False),
            ).props("flat color=primary").classes("self-start mt-auto mb-2")

        toolbar_toggle = ui.element("div").classes(
            "fixed left-0 top-1/2 -translate-y-1/2 z-40 cursor-pointer "
            "bg-primary text-white rounded-r-lg shadow-lg opacity-70 hover:opacity-100 "
            "transition-all duration-300 px-1 py-3"
        ).style("min-width:16px;")
        toolbar_toggle.on("click", lambda _: config_toolbar.set_value(True))
        with toolbar_toggle:
            ui.icon("chevron_right").classes("text-white text-lg")

        with ui.column().classes(PAGE_CONTAINER_CLASSES):
            ui.markdown("## DILIGENT Clinical Copilot").classes(
                "text-3xl font-semibold text-slate-800 dark:text-slate-100"
            )

            with ui.element("div").classes("diligent-equal-row"):
                with ui.column().classes("diligent-equal-column flex-1 w-full"):
                    with ui.card().classes(
                        f"{CARD_BASE_CLASSES} diligent-equal-card w-full"
                    ):
                        ui.label("Clinical Inputs").classes("diligent-card-title")
                        anamnesis = ui.textarea(
                            label="Anamnesis",
                            placeholder=(
                                "Describe the clinical picture, including exams and labs when relevant..."
                            ),
                        ).classes("w-full min-h-[12rem]")
                        drugs = ui.textarea(
                            label="Current Drugs",
                            placeholder="List current therapies, dosage and schedule...",
                        ).classes("w-full min-h-[12rem]")
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            alt = ui.input(
                                label="ALT",
                                placeholder="e.g., 189 or 189 U/L",
                            ).classes("w-full")
                            alt.props("dense")
                            alt_max = ui.input(
                                label="ALT Max",
                                placeholder="e.g., 47 U/L",
                            ).classes("w-full")
                            alt_max.props("dense")
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            alp = ui.input(
                                label="ALP",
                                placeholder="e.g., 140 or 140 U/L",
                            ).classes("w-full")
                            alp.props("dense")
                            alp_max = ui.input(
                                label="ALP Max",
                                placeholder="e.g., 150 U/L",
                            ).classes("w-full")
                            alp_max.props("dense")

                with ui.column().classes("diligent-equal-column w-full"):
                    with ui.card().classes(
                        f"{CARD_BASE_CLASSES} diligent-equal-card w-full"
                    ):
                        ui.label("Patient Information").classes(
                            "diligent-card-title"
                        )
                        patient_name = ui.input(
                            label="Patient Name",
                            placeholder="e.g., Marco Rossi",
                        ).classes("w-full")
                        with ui.element("div").props(f"id={VISIT_DATE_ELEMENT_ID}"):
                            visit_date = ui.input(
                                label="Visit Date",
                                placeholder="Select a date",
                            ).classes("w-full")
                        visit_date.props("type=date")
                        with ui.column().classes("w-full gap-3"):
                            run_button = ui.button(
                                "Run DILI analysis", color="primary"
                            ).classes("w-full")
                            export_button = ui.button(
                                "Download report", color="secondary"
                            ).classes("w-full")
                            export_button.disable()
                            clear_button = ui.button("Clear all").classes("w-full")
                        spinner_container = ui.element("div").classes(
                            "diligent-session-spinner-container"
                        )
                        with spinner_container:
                            session_spinner = ui.element("div").classes(
                                "diligent-session-spinner"
                            )
                            with session_spinner:
                                ui.element("div").classes("diligent-spinner-wheel")
                                ui.label("Generating report...").classes(
                                    "diligent-spinner-label"
                                )
                        session_spinner.visible = False

            with ui.card().classes(f"{CARD_BASE_CLASSES} w-full"):
                ui.label("Report Output").classes("diligent-card-title")
                markdown_output = ui.markdown("").classes(
                    "text-sm whitespace-pre-wrap"
                )

            with ui.card().classes(JSON_CARD_CLASSES) as json_container:
                json_container.classes("w-full")
                with json_container:
                    ui.label("JSON Response").classes("diligent-subtitle")
                    json_code = ui.code("", language="json").classes("w-full")

        setattr(export_button, export_attribute, None)
        self.toolkit.update_cloud_controls(
            cloud_enabled,
            llm_provider_dropdown,
            cloud_model_dropdown,
            parsing_model_dropdown,
            clinical_model_dropdown,
            temperature_input,
            reasoning_checkbox,
            pull_models_button,
        )

        use_cloud_services.on_value_change(
            partial(
                self.controller.handle_toggle_cloud_services,
                llm_provider_dropdown,
                cloud_model_dropdown,
                parsing_model_dropdown,
                clinical_model_dropdown,
                temperature_input,
                reasoning_checkbox,
                pull_models_button,
            )
        )

        llm_provider_dropdown.on_value_change(
            partial(
                self.controller.handle_cloud_provider_change,
                llm_provider_dropdown,
                cloud_model_dropdown,
            )
        )

        pull_models_button.on(
            "click",
            partial(
                self.controller.handle_pull_models_click,
                use_cloud_services,
                llm_provider_dropdown,
                cloud_model_dropdown,
                parsing_model_dropdown,
                clinical_model_dropdown,
                temperature_input,
                reasoning_checkbox,
                markdown_output,
                json_container,
                json_code,
            ),
        )

        run_button.on(
            "click",
            partial(
                self.controller.handle_run_workflow,
                patient_name,
                visit_date,
                anamnesis,
                drugs,
                alt,
                alt_max,
                alp,
                alp_max,
                use_rag_checkbox,
                use_cloud_services,
                llm_provider_dropdown,
                cloud_model_dropdown,
                parsing_model_dropdown,
                clinical_model_dropdown,
                temperature_input,
                reasoning_checkbox,
                markdown_output,
                session_spinner,
                json_container,
                json_code,
                export_button,
            ),
        )

        clear_button.on(
            "click",
            partial(
                self.controller.handle_clear_click,
                patient_name,
                visit_date,
                anamnesis,
                drugs,
                alt,
                alt_max,
                alp,
                alp_max,
                use_rag_checkbox,
                use_cloud_services,
                llm_provider_dropdown,
                cloud_model_dropdown,
                pull_models_button,
                parsing_model_dropdown,
                clinical_model_dropdown,
                temperature_input,
                reasoning_checkbox,
                markdown_output,
                session_spinner,
                json_container,
                json_code,
                export_button,
            ),
        )

        visit_date.on_value_change(
            partial(self.controller.handle_visit_date_change, visit_date)
        )

        export_button.on(
            "click", partial(self.controller.handle_download_click, export_button)
        )

        ui.run_javascript(f"({VISIT_DATE_LOCALE_JS})();")

    # -------------------------------------------------------------------------
    def mount_routes(self) -> None:
        ui.page("/")(self.compose_main_page)
        

# [INTERFACE CREATION AND LAUNCHING]
###############################################################################
def create_interface() -> InterfaceStructure: 
    settings_controller = SettingsService()
    model_pull_controller = ModelPullEndpointService()
    dili_session_controller = DILISessionEndpointService()
    toolkit = InterfaceToolkit()
    controller = InterfaceService(
        settings_controller,
        model_pull_controller,
        dili_session_controller,
        toolkit
    )
    structure = InterfaceStructure(
        settings_controller, 
        controller,
        toolkit)
    
    structure.mount_routes()

    return structure

# -----------------------------------------------------------------------------
def launch_interface() -> None:
    create_interface()
    ui.run(
        host=ui_settings.host,
        port=ui_settings.port,
        title=ui_settings.title,
        show_welcome_message=ui_settings.show_welcome_message,
        reconnect_timeout=ui_settings.reconnect_timeout,
    )

# -----------------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:
    launch_interface()
