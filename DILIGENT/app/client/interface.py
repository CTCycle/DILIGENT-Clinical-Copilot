from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Any

from nicegui import ui

from DILIGENT.app.client.controllers import (
    ComponentUpdate,
    MISSING,
    clear_session_fields,
    normalize_visit_date_component,
    pull_selected_models,
    run_DILI_session,
    set_clinical_model,
    set_cloud_model,
    set_llm_provider,
    set_ollama_reasoning,
    set_ollama_temperature,
    set_parsing_model,
    toggle_cloud_services,
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
from DILIGENT.app.configurations import (ClientRuntimeConfig, CLOUD_PROVIDERS)
from DILIGENT.app.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    PARSING_MODEL_CHOICES,
)


###############################################################################
@dataclass
class ClientComponents:
    patient_name: Any
    visit_date: Any
    anamnesis: Any
    has_diseases: Any
    drugs: Any
    alt: Any
    alt_max: Any
    alp: Any
    alp_max: Any
    use_rag: Any
    use_cloud_services: Any
    llm_provider_dropdown: Any
    cloud_model_dropdown: Any
    parsing_model_dropdown: Any
    clinical_model_dropdown: Any
    temperature_input: Any
    reasoning_checkbox: Any
    pull_models_button: Any
    markdown_output: Any
    json_container: Any
    json_code: Any
    export_button: Any
    export_path: str | None = None


###############################################################################
def apply_component_update(component: Any, update: ComponentUpdate) -> None:
    if update.value is not MISSING and hasattr(component, "value"):
        value_to_set = update.value
        if value_to_set == "" and hasattr(component, "set_options"):
            value_to_set = None
        component.value = value_to_set
        if hasattr(component, "update"):
            component.update()
    if update.options is not None and hasattr(component, "set_options"):
        component.set_options(update.options)
        if hasattr(component, "update"):
            component.update()
    if update.enabled is not None:
        if update.enabled and hasattr(component, "enable"):
            component.enable()
        elif not update.enabled and hasattr(component, "disable"):
            component.disable()
    if update.visible is not None and hasattr(component, "visible"):
        component.visible = update.visible


###############################################################################
def apply_json_update(components: ClientComponents, update: ComponentUpdate) -> None:
    if update.value is not MISSING:
        if update.value is None:
            components.json_code.set_content("")
        else:
            formatted = json.dumps(update.value, ensure_ascii=False, indent=2)
            components.json_code.set_content(formatted)
    if update.visible is not None:
        components.json_container.visible = update.visible


###############################################################################
def apply_export_update(components: ClientComponents, update: ComponentUpdate) -> None:
    if update.download_path is not None:
        components.export_path = update.download_path
    elif update.value is not MISSING:
        export_value = update.value
        components.export_path = export_value if isinstance(export_value, str) else None
    if update.enabled is not None:
        if update.enabled:
            components.export_button.enable()
        else:
            components.export_button.disable()


###############################################################################
async def handle_toggle_cloud_services(
    components: ClientComponents, event: Any
) -> None:
    enabled = bool(event.value)
    updates = toggle_cloud_services(enabled)
    apply_component_update(components.llm_provider_dropdown, updates[0])
    apply_component_update(components.cloud_model_dropdown, updates[1])
    apply_component_update(components.pull_models_button, updates[2])
    apply_component_update(components.temperature_input, updates[3])
    apply_component_update(components.reasoning_checkbox, updates[4])
    apply_component_update(components.clinical_model_dropdown, updates[5])


###############################################################################
async def handle_llm_provider_change(components: ClientComponents, event: Any) -> None:
    provider_value = str(event.value or "")
    selected, model_update = set_llm_provider(provider_value)
    if hasattr(components.llm_provider_dropdown, "value"):
        components.llm_provider_dropdown.value = selected
        components.llm_provider_dropdown.update()
    apply_component_update(components.cloud_model_dropdown, model_update)


###############################################################################
async def handle_cloud_model_change(components: ClientComponents, event: Any) -> None:
    selection = set_cloud_model(str(event.value or ""))
    if hasattr(components.cloud_model_dropdown, "value"):
        components.cloud_model_dropdown.value = selection
        components.cloud_model_dropdown.update()


###############################################################################
async def handle_parsing_model_change(components: ClientComponents, event: Any) -> None:
    selection = set_parsing_model(str(event.value or ""))
    if hasattr(components.parsing_model_dropdown, "value"):
        components.parsing_model_dropdown.value = selection
        components.parsing_model_dropdown.update()


###############################################################################
async def handle_clinical_model_change(
    components: ClientComponents, event: Any
) -> None:
    selection = set_clinical_model(str(event.value or ""))
    if hasattr(components.clinical_model_dropdown, "value"):
        components.clinical_model_dropdown.value = selection
        components.clinical_model_dropdown.update()


###############################################################################
async def handle_temperature_change(components: ClientComponents, event: Any) -> None:
    value = event.value if event.value is not None else None
    updated = set_ollama_temperature(value)
    if hasattr(components.temperature_input, "value"):
        components.temperature_input.value = updated
        components.temperature_input.update()


###############################################################################
async def handle_reasoning_change(components: ClientComponents, event: Any) -> None:
    updated = set_ollama_reasoning(bool(event.value))
    if hasattr(components.reasoning_checkbox, "value"):
        components.reasoning_checkbox.value = updated
        components.reasoning_checkbox.update()


###############################################################################
async def handle_pull_models_click(components: ClientComponents, event: Any) -> None:
    parsing_model = str(components.parsing_model_dropdown.value or "")
    clinical_model = str(components.clinical_model_dropdown.value or "")
    message, json_update = await pull_selected_models(parsing_model, clinical_model)
    components.markdown_output.set_content(message or "")
    apply_json_update(components, json_update)


###############################################################################
async def handle_run_workflow(components: ClientComponents, event: Any) -> None:
    message, json_update, export_update = await run_DILI_session(
        components.patient_name.value,
        components.visit_date.value,
        components.anamnesis.value,
        bool(components.has_diseases.value),
        components.drugs.value,
        components.alt.value,
        components.alt_max.value,
        components.alp.value,
        components.alp_max.value,
        bool(components.use_rag.value),
    )
    components.markdown_output.set_content(message or "")
    apply_json_update(components, json_update)
    apply_export_update(components, export_update)


###############################################################################
async def handle_clear_click(components: ClientComponents, event: Any) -> None:
    (
        patient_name,
        visit_date,
        anamnesis,
        drugs,
        alt,
        alt_max,
        alp,
        alp_max,
        has_diseases,
        use_rag,
        markdown_message,
        json_update,
        export_update,
    ) = clear_session_fields()
    components.patient_name.value = patient_name
    components.patient_name.update()
    components.visit_date.value = visit_date or ""
    components.visit_date.update()
    components.anamnesis.value = anamnesis
    components.anamnesis.update()
    components.drugs.value = drugs
    components.drugs.update()
    components.alt.value = alt
    components.alt.update()
    components.alt_max.value = alt_max
    components.alt_max.update()
    components.alp.value = alp
    components.alp.update()
    components.alp_max.value = alp_max
    components.alp_max.update()
    components.has_diseases.value = has_diseases
    components.has_diseases.update()
    components.use_rag.value = use_rag
    components.use_rag.update()
    components.markdown_output.set_content(markdown_message or "")
    apply_json_update(components, json_update)
    apply_export_update(components, export_update)


###############################################################################
async def handle_visit_date_change(components: ClientComponents, event: Any) -> None:
    normalized = normalize_visit_date_component(event.value)
    if normalized is None:
        components.visit_date.value = ""
    else:
        components.visit_date.value = normalized.date().isoformat()
    components.visit_date.update()


###############################################################################
async def handle_download_click(components: ClientComponents, event: Any) -> None:
    if components.export_path:
        ui.download(components.export_path)


###############################################################################
def main_page() -> None:
    provider = ClientRuntimeConfig.get_llm_provider()
    cloud_models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_cloud_model = ClientRuntimeConfig.get_cloud_model()
    if selected_cloud_model not in cloud_models:
        selected_cloud_model = cloud_models[0] if cloud_models else ""
        ClientRuntimeConfig.set_cloud_model(selected_cloud_model)

    cloud_enabled = ClientRuntimeConfig.is_cloud_enabled()

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
                    has_diseases = ui.checkbox(
                        "Has hepatic diseases",
                        value=False,
                    ).classes("pt-2")
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
                                value=ClientRuntimeConfig.get_parsing_model(),
                            ).classes("w-full")
                            clinical_model_dropdown = ui.select(
                                CLINICAL_MODEL_CHOICES,
                                label="Clinical Model",
                                value=ClientRuntimeConfig.get_clinical_model(),
                            ).classes("w-full")
                            temperature_input = ui.number(
                                label="Temperature",
                                value=ClientRuntimeConfig.get_ollama_temperature(),
                                min=0.0,
                                max=2.0,
                                step=0.1,
                            ).classes("w-full")
                            reasoning_checkbox = ui.checkbox(
                                "Enable reasoning (think)",
                                value=ClientRuntimeConfig.is_ollama_reasoning_enabled(),
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

    components = ClientComponents(
        patient_name=patient_name,
        visit_date=visit_date,
        anamnesis=anamnesis,
        has_diseases=has_diseases,
        drugs=drugs,
        alt=alt,
        alt_max=alt_max,
        alp=alp,
        alp_max=alp_max,
        use_rag=use_rag_checkbox,
        use_cloud_services=use_cloud_services,
        llm_provider_dropdown=llm_provider_dropdown,
        cloud_model_dropdown=cloud_model_dropdown,
        parsing_model_dropdown=parsing_model_dropdown,
        clinical_model_dropdown=clinical_model_dropdown,
        temperature_input=temperature_input,
        reasoning_checkbox=reasoning_checkbox,
        pull_models_button=pull_models_button,
        markdown_output=markdown_output,
        json_container=json_container,
        json_code=json_code,
        export_button=export_button,
    )

    if cloud_enabled:
        llm_provider_dropdown.enable()
        cloud_model_dropdown.enable()
        pull_models_button.disable()
        temperature_input.disable()
        reasoning_checkbox.disable()
        clinical_model_dropdown.disable()
    else:
        llm_provider_dropdown.disable()
        cloud_model_dropdown.disable()
        pull_models_button.enable()
        temperature_input.enable()
        reasoning_checkbox.enable()
        clinical_model_dropdown.enable()

    use_cloud_services.on_value_change(
        partial(handle_toggle_cloud_services, components)
    )
    llm_provider_dropdown.on_value_change(
        partial(handle_llm_provider_change, components)
    )
    cloud_model_dropdown.on_value_change(partial(handle_cloud_model_change, components))
    parsing_model_dropdown.on_value_change(
        partial(handle_parsing_model_change, components)
    )
    clinical_model_dropdown.on_value_change(
        partial(handle_clinical_model_change, components)
    )
    temperature_input.on_value_change(partial(handle_temperature_change, components))
    reasoning_checkbox.on_value_change(partial(handle_reasoning_change, components))
    pull_models_button.on("click", partial(handle_pull_models_click, components))
    run_button.on("click", partial(handle_run_workflow, components))
    clear_button.on("click", partial(handle_clear_click, components))
    visit_date.on_value_change(partial(handle_visit_date_change, components))
    export_button.on("click", partial(handle_download_click, components))

    ui.run_javascript(f"({VISIT_DATE_LOCALE_JS})();")


###############################################################################
def create_interface() -> None:
    ui.page("/")(main_page)


###############################################################################
def launch_interface() -> None:
    create_interface()
    ui.run(
        host="0.0.0.0",
        port=7861,
        title="DILIGENT Clinical Copilot",
        show_welcome_message=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    launch_interface()
