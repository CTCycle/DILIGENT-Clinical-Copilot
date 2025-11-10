from __future__ import annotations

import json
from functools import partial
from typing import Any

from nicegui import ui

from DILIGENT.app.client.controllers import (
    RuntimeSettings,
    clear_session_fields,
    get_runtime_settings,
    normalize_visit_date_component,
    pull_selected_models,
    resolve_cloud_selection,
    run_DILI_session,    
)
from DILIGENT.app.client.layouts import (
    CARD_BASE_CLASSES,
    INTERFACE_THEME_CSS,
    JSON_CARD_CLASSES,
    PAGE_CONTAINER_CLASSES,
    VISIT_DATE_CSS,
    VISIT_DATE_ELEMENT_ID,
    VISIT_DATE_LOCALE_JS,
)
from DILIGENT.app.configurations import UI_RUNTIME_SETTINGS
from DILIGENT.app.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    PARSING_MODEL_CHOICES,
)

export_attribute = "export_path"

CLOUD_PROVIDERS: list[str] = [key for key in CLOUD_MODEL_CHOICES]


# HELPERS
###############################################################################
def update_json_display(container: Any, code: Any, payload: Any) -> None:
    if payload is not None:
        container.visible = True
        formatted = json.dumps(payload, ensure_ascii=False, indent=2)
        code.set_content(formatted)
    else:
        container.visible = False
        code.set_content("")

# -----------------------------------------------------------------------------
def update_cloud_controls(
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

# -----------------------------------------------------------------------------
async def handle_toggle_cloud_services(
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
    selection = resolve_cloud_selection(
        str(llm_provider_dropdown.value or ""),
        str(cloud_model_dropdown.value or ""),
    )
    llm_provider_dropdown.value = selection["provider"]
    llm_provider_dropdown.update()
    cloud_model_dropdown.set_options(selection["models"])
    cloud_model_dropdown.value = selection["model"]
    cloud_model_dropdown.update()
    update_cloud_controls(
        enabled,
        llm_provider_dropdown,
        cloud_model_dropdown,
        parsing_model_dropdown,
        clinical_model_dropdown,
        temperature_input,
        reasoning_checkbox,
        pull_models_button,
    )

# -----------------------------------------------------------------------------
async def handle_cloud_provider_change(
    llm_provider_dropdown: Any,
    cloud_model_dropdown: Any,
    event: Any,
) -> None:
    selection = resolve_cloud_selection(
        str(event.value or ""),
        str(cloud_model_dropdown.value or ""),
    )
    llm_provider_dropdown.value = selection["provider"]
    llm_provider_dropdown.update()
    cloud_model_dropdown.set_options(selection["models"])
    cloud_model_dropdown.value = selection["model"]
    cloud_model_dropdown.update()

# -----------------------------------------------------------------------------
async def handle_visit_date_change(visit_date_input: Any, event: Any) -> None:
    normalized = normalize_visit_date_component(event.value)
    visit_date_input.value = "" if normalized is None else normalized.date().isoformat()
    visit_date_input.update()
    

# ACTIONS
###############################################################################
async def handle_pull_models_click(
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
    result = await pull_selected_models(settings)
    markdown_output.set_content(result.get("message") or "")
    update_json_display(json_container, json_code, result.get("json"))

# -----------------------------------------------------------------------------
async def handle_run_workflow(
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
    result = await run_DILI_session(
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
    markdown_output.set_content(result.get("message") or "")
    update_json_display(json_container, json_code, result.get("json"))
    export_path = result.get(export_attribute)
    setattr(export_button, export_attribute, export_path)
    if export_path:
        export_button.enable()
    else:
        export_button.disable()

# -----------------------------------------------------------------------------
async def handle_clear_click(
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
    json_container: Any,
    json_code: Any,
    export_button: Any,
    event: Any,
) -> None:
    defaults = clear_session_fields()
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
    selection = resolve_cloud_selection(settings.provider, settings.cloud_model)
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
    update_cloud_controls(
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
    update_json_display(json_container, json_code, defaults.get("json"))
    setattr(export_button, export_attribute, defaults.get(export_attribute))
    export_button.disable()

# -----------------------------------------------------------------------------
async def handle_download_click(export_button: Any, event: Any) -> None:
    export_path = getattr(export_button, export_attribute, None)
    if export_path:
        ui.download(export_path)

###############################################################################
# MAIN UI PAGE
###############################################################################
def main_page() -> None:
    current_settings = get_runtime_settings()
    selection = resolve_cloud_selection(
        current_settings.provider, current_settings.cloud_model
    )
    provider = selection["provider"]
    cloud_models = selection["models"]
    selected_cloud_model = selection["model"]
    cloud_enabled = current_settings.use_cloud_services

    ui.page_title("DILIGENT Clinical Copilots")
    ui.add_head_html(f"<style>{VISIT_DATE_CSS}{INTERFACE_THEME_CSS}</style>")

    with ui.column().classes(PAGE_CONTAINER_CLASSES):
        ui.markdown("## DILIGENT Clinical Copilot").classes(
            "text-3xl font-semibold text-slate-800 dark:text-slate-100"
        )

        with ui.row().classes("w-full gap-6 flex-col xl:flex-row"):
            with ui.column().classes("flex-1 w-full gap-6"):
                with ui.card().classes(f"{CARD_BASE_CLASSES} w-full"):
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

            with ui.column().classes("w-full xl:w-[26rem] gap-6"):
                with ui.card().classes(f"{CARD_BASE_CLASSES} w-full"):
                    ui.label("Patient Information").classes("diligent-card-title")
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

                with ui.expansion("Models & Analysis Configuration").classes("w-full"):
                    ui.label("Configuration").classes("diligent-card-title")
                    use_rag_checkbox = ui.checkbox(
                        "Use Retrieval Augmented Generation (RAG)",
                        value=False,
                    ).classes("pt-2")
                    use_cloud_services = ui.checkbox(
                        "Use Cloud Services",
                        value=cloud_enabled,
                    ).classes("pt-2")
                    with ui.grid(columns=1).classes("w-full gap-5 lg:grid-cols-2"):
                        with ui.column().classes("w-full gap-3"):
                            ui.label("Cloud Configuration").classes("diligent-subtitle")
                            llm_provider_dropdown = ui.select(
                                CLOUD_PROVIDERS,
                                label="Cloud Service",
                                value=provider,
                            ).classes("w-full")
                            cloud_model_dropdown = ui.select(
                                cloud_models,
                                label="Cloud Model",
                                value=selected_cloud_model or None,
                            ).classes("w-full")
                        with ui.column().classes("w-full gap-3"):
                            ui.label("Ollama Configuration").classes(
                                "diligent-subtitle"
                            )
                            parsing_model_dropdown = ui.select(
                                PARSING_MODEL_CHOICES,
                                label="Parsing Model",
                                value=current_settings.parsing_model,
                            ).classes("w-full")
                            clinical_model_dropdown = ui.select(
                                CLINICAL_MODEL_CHOICES,
                                label="Clinical Model",
                                value=current_settings.clinical_model,
                            ).classes("w-full")
                            temperature_input = ui.number(
                                label="Temperature",
                                value=current_settings.temperature,
                                min=0.0,
                                max=5.0,
                                step=0.1,
                            ).classes("w-full")
                            reasoning_checkbox = ui.checkbox(
                                "Enable reasoning (think)",
                                value=current_settings.reasoning,
                            )
                            pull_models_button = ui.button(
                                "Pull models", color="secondary"
                            )

        with ui.card().classes(f"{CARD_BASE_CLASSES} w-full"):
            ui.label("Results & Reports").classes("diligent-card-title")
            with ui.row().classes("w-full gap-4 flex-col lg:flex-row"):
                with ui.column().classes("w-full gap-3"):
                    ui.label("Agent Output").classes("diligent-subtitle")
                    markdown_output = ui.markdown("").classes(
                        "prose prose-slate max-w-none dark:prose-invert"
                    )
                with ui.column().classes("w-full gap-3"):
                    json_container = ui.card().classes(JSON_CARD_CLASSES)
                    json_container.visible = False
                    with json_container:
                        ui.label("JSON Response").classes("diligent-subtitle")
                        json_code = ui.code("", language="json").classes("w-full")

    setattr(export_button, export_attribute, None)

    update_cloud_controls(
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
            handle_toggle_cloud_services,
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
            handle_cloud_provider_change, 
            llm_provider_dropdown, 
            cloud_model_dropdown)
    )
    pull_models_button.on(
        "click",
        partial(
            handle_pull_models_click,
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
            handle_run_workflow,
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
            json_container,
            json_code,
            export_button,
        ),
    )
    clear_button.on(
        "click",
        partial(
            handle_clear_click,
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
            json_container,
            json_code,
            export_button,
        ),
    )
    visit_date.on_value_change(partial(handle_visit_date_change, visit_date))
    export_button.on("click", partial(handle_download_click, export_button))

    ui.run_javascript(f"({VISIT_DATE_LOCALE_JS})();")


# MOUNT AND LAUNCH INTERFACE
###############################################################################
def create_interface() -> None:
    ui.page("/")(main_page)

# -----------------------------------------------------------------------------
def launch_interface() -> None:
    create_interface()
    ui.run(
        host=UI_RUNTIME_SETTINGS.host,
        port=UI_RUNTIME_SETTINGS.port,
        title=UI_RUNTIME_SETTINGS.title,
        show_welcome_message=UI_RUNTIME_SETTINGS.show_welcome_message,
    )

# -----------------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:
    launch_interface()
