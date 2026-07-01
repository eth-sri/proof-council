from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EditorTemplateContractTests(unittest.TestCase):
    def test_home_page_renders_favicon_above_title(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_home.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertLess(template.index("home-logo"), template.index("<h1>ProofCouncil</h1>"))
        self.assertIn("filename='favicon.png'", template)
        self.assertIn(".home-logo", base)
        self.assertIn("object-fit: contain;", base)


    def test_run_agent_exposes_monitor_controls(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_run_agent.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertIn("run-agent-monitor-enabled", template)
        self.assertIn("run-agent-monitor-model", template)
        self.assertIn("monitor_model_options", template)
        self.assertIn("monitorKeyRequirements", template)
        self.assertIn("function activeApiKeyRequirements", template)
        self.assertIn("monitor: document.getElementById('run-agent-monitor-enabled')", template)
        self.assertIn("monitor_model: document.getElementById('run-agent-monitor-model')", template)
        self.assertIn('.run-agent-field input[type="checkbox"]', base)

    def test_editor_sample_run_exposes_monitor_controls(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertIn("sample-monitor-enabled", template)
        self.assertIn("sample-monitor-model", template)
        self.assertIn("monitor_model_options", template)
        self.assertIn("monitor: document.getElementById('sample-monitor-enabled')", template)
        self.assertIn("monitor_model: document.getElementById('sample-monitor-model')", template)
        self.assertIn(".sample-runner select", base)

    def test_run_detail_has_live_monitor_section(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_run_detail.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertIn("run-monitor-section", template)
        self.assertIn("dev_run_monitor.html", template)
        self.assertIn("startRunMonitorPolling", template)
        self.assertLess(template.index("Workflow input"), template.index("Monitor summaries"))
        self.assertLess(template.index("Monitor summaries"), template.index("Workflow output"))
        self.assertIn("function startRunMonitorPolling", base)
        self.assertIn(".monitor-summary", base)

    def test_workflow_output_edges_from_repeat_nodes_are_mapped_before_output_target(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function buildCanvasEdges")
        end = template.index("function repeatZoneInternalEdges", start)
        body = template[start:end]

        source_map = body.index("loopIds.has(mapped.source)")
        output_map = body.index("mapped.target === 'outputs'")

        self.assertLess(source_map, output_map)
        self.assertIn("mapped.source = repeatOutputNodeId(mapped.source)", body)
        self.assertIn("mapped.target = workflowOutputNodeId", body)

    def test_prompt_sharing_renders_before_tools_in_prompt_editor(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function componentForm")
        end = template.index("function defaultPromptInputSource", start)
        body = template[start:end]

        sharing_index = body.index("promptTiePanel(name, spec, prefix)")
        tools_index = body.index("componentToolControls(cfg, prefix)")

        self.assertLess(sharing_index, tools_index)

    def test_cli_agent_prompt_editor_exposes_cli_model_and_reasoning_effort(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function componentForm")
        end = template.index("function defaultPromptInputSource", start)
        body = template[start:end]

        self.assertIn("CLI model", body)
        self.assertIn("cliModelControl(prefix, cfg)", body)
        self.assertIn("Reasoning effort", body)
        self.assertIn("cliReasoningEffortSelect(prefix, cliReasoningEffortValue(cfg))", body)
        self.assertIn("model_reasoning_effort", template)
        self.assertIn("cliModelFromCmd", template)
        self.assertIn("cliReasoningEffortFromCmd", template)

    def test_repeat_zone_only_draws_editable_internal_wires(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function repeatZoneInternalEdges")
        end = template.index("function canvasEdge", start)
        body = template[start:end]

        self.assertNotIn("canvasEdge('dependency'", body)
        self.assertIn("canvasEdge('data'", body)
        self.assertIn("canvasEdge('repeat_state'", body)
        self.assertIn("canvasEdge('repeat_update'", body)
        self.assertIn("isIdentityRepeatCarryRef(field, ref)", body)
        self.assertNotIn("canvasEdge('repeat_update', inputId, field", body)

    def test_repeat_state_edges_with_raw_target_paths_are_editable(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function edgeTargetField")
        end = template.index("function edgeTransform", start)
        body = template[start:end]

        self.assertIn("['data', 'repeat_state', 'repeat_update'].includes(edge.edge_kind)", body)
        self.assertIn("return topPathSegment(path);", body)

    def test_repeat_creation_uses_canvas_node_picker(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function renderRepeatZoneCreatePanel")
        end = template.index("function repeatZoneCaptureSection", start)
        body = template[start:end]

        self.assertIn("repeatPathPicker(prefix", body)
        self.assertIn("handleRepeatNodePickClick(node.id, event)", template)
        self.assertIn("beginRepeatNodePick", template)
        self.assertNotIn("<select id=\"${prefix}-start\"", body)
        self.assertNotIn("topLevelRepeatCandidateOptions", template)
        self.assertIn(".canvas-node.repeat-pick-candidate", base)

    def test_repeat_memory_nodes_are_movable_and_skip_runtime_fields(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function repeatZoneVisualNodes")
        end = template.index("function renderRepeatZoneBackdrops", start)
        repeat_body = template[start:end]

        self.assertIn("repeatVirtualNodePosition(loopNode, 'repeat_input'", repeat_body)
        self.assertIn("repeatVirtualNodePosition(loopNode, 'repeat_output'", repeat_body)
        self.assertIn("input_fields: repeatItems", repeat_body)
        self.assertNotIn("input_fields: ['iterations', ...repeatItems]", repeat_body)
        self.assertIn("function isRepeatRuntimeField", repeat_body)
        self.assertIn("'next_iteration'", repeat_body)
        self.assertIn("move_repeat_virtual_node", template)
        self.assertIn("startRepeatZoneDrag(event, loopNode.id)", template)
        self.assertIn("function startRepeatZoneDrag", template)
        self.assertIn("function repeatZoneVisualMembers", template)
        self.assertIn("op: 'move_repeat_zone_visuals'", template)
        self.assertIn("function repeatZoneMoveOperation", template)
        self.assertIn("['loop_body', 'repeat_input', 'repeat_output'].includes", template)

    def test_repeat_zone_backdrop_can_receive_pointer_events_beneath_node_layer(self) -> None:
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        node_layer = base[base.index(".node-layer {"):base.index(".repeat-zone-backdrop", base.index(".node-layer {"))]
        canvas_node = base[base.index(".canvas-node {"):base.index(".canvas-node.virtual-node", base.index(".canvas-node {"))]
        repeat_backdrop = base[base.index(".repeat-zone-backdrop {"):base.index(".repeat-zone-backdrop:active", base.index(".repeat-zone-backdrop {"))]

        self.assertIn("pointer-events: none;", node_layer)
        self.assertIn("pointer-events: auto;", canvas_node)
        self.assertIn("pointer-events: auto;", repeat_backdrop)

    def test_add_node_palette_exposes_exported_presets_directly(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        palette_start = template.index("function addNodePaletteItems")
        palette_end = template.index("function openAddNodePalette", palette_start)
        palette_body = template[palette_start:palette_end]
        add_start = template.index("async function addGraphNode")
        add_end = template.index("async function pasteCopiedGraphNode", add_start)
        add_body = template[add_start:add_end]

        self.assertNotIn("label: 'Subagent Preset'", template)
        self.assertIn("exportedPresets", palette_body)
        self.assertIn("group: 'Subagents'", palette_body)
        self.assertIn("agentPaletteItems", palette_body)
        self.assertIn("presetInfo?.label", add_body)
        self.assertIn("node_id: template === 'workflow_ref'", add_body)
        self.assertIn("template === 'python_agent'", add_body)
        self.assertNotIn("template: 'parallel_svi'", template)
        for obsolete in (
            "template: 'solver'",
            "template: 'validator'",
            "template: 'improver'",
            "template: 'map_chain'",
            "template: 'join'",
            "Proof Writer",
            "Checker",
            "Fixer",
            "Parallel Proof Attempts",
            "Merge Proofs",
        ):
            self.assertNotIn(obsolete, template)

    def test_repeat_max_iterations_input_can_show_expressions(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function nodeLoopSection")
        end = template.index("function repeatStateEditor", start)
        body = template[start:end]

        self.assertIn("repeatMaxIterationsInputAttrs", body)
        self.assertIn("Object.prototype.hasOwnProperty.call(cfg, 'max_iterations')", body)
        self.assertIn("type=\"text\" spellcheck=\"false\"", template)
        self.assertNotIn('type="number" min="1" value="${escapeAttr(editorValue(cfg.max_iterations || 3))}', body)

    def test_inspector_sections_preserve_user_open_state(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        self.assertIn("const inspectorSectionOpenStates = new Map();", template)
        self.assertIn("data-inspector-section-key", template)
        self.assertIn("handleInspectorSectionToggle", template)
        self.assertIn("section: 'outputs'", template)

    def test_if_else_nodes_render_branch_output_editor(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function nodeOutputEditor")
        end = template.index("function nodeOutputSection", start)
        body = template[start:end]

        self.assertIn("node.kind === 'if_else'", body)
        self.assertIn("ifElseOutputEditor(node, prefix)", body)
        self.assertIn("True output", template)
        self.assertIn("False output", template)
        self.assertIn("scheduleIfOutputsAutosave", template)
        self.assertNotIn("Output descriptions", template)
        self.assertNotIn("class=\"node-output-editor if-output-editor", template)

    def test_cli_agent_outputs_edit_workspace_file_outputs(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function nodeOutputEditor")
        end = template.index("function ifElseOutputEditor", start)
        body = template[start:end]

        self.assertIn("node.agent === configurableCliAgent", body)
        self.assertIn("cliOutputFilesEditor(node, cfg, prefix, componentName)", body)
        self.assertIn("Workspace file outputs", template)
        self.assertIn("Add file output", template)
        self.assertIn("output_files: outputFiles", template)
        self.assertIn("output_schema: outputSchema", template)
        self.assertIn("cliOutputFieldForPath(path)", template)
        self.assertIn("placeholder=\"answers.tex\"", template)
        self.assertNotIn("cli-output-type", template)
        self.assertNotIn("cli-output-default", template)
        self.assertNotIn("default if missing", template)

    def test_cli_agent_output_autosave_does_not_rerender_inspector(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function scheduleCliOutputFilesAutosaveFromElement")
        end = template.index("function hasIncompleteCliOutputFileRow", start)
        body = template[start:end]

        self.assertIn("applyCliOutputFilesWithPrefix(name, prefix, {", body)
        self.assertIn("renderReport: false", body)
        self.assertIn("renderDagCanvas({preserveViewport: true})", template)
        self.assertIn("__rename_output_refs", template)
        self.assertIn("markCliOutputFilesSaved(prefix)", template)

    def test_if_branch_wires_become_run_conditions(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("async function finishConnectionDrag")
        end = template.index("function cancelConnectionDrag", start)
        finish_body = template[start:end]

        self.assertIn("function branchAdjustedTargetField", template)
        self.assertIn("return '__condition';", template)
        self.assertIn("canvas-node-drop-target", template)
        self.assertIn(".canvas-node.canvas-node-drop-target", base)
        self.assertIn('inputs.get(\\"False\\")', template)
        self.assertLess(finish_body.index("const targetField"), finish_body.index("cancelConnectionDrag()"))

    def test_run_condition_section_has_visible_help_and_replaces_low_level(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function conditionUpdatePayload")
        end = template.index("async function applyNodeDefaultOutputs", start)
        payload_body = template[start:end]

        self.assertIn("inspectorSection('Run condition'", template)
        self.assertNotIn("inspectorSection('Low-Level'", template)
        self.assertIn("Reference / Condition", template)
        self.assertIn("fieldHelp.conditionReference", template)
        self.assertIn("fieldHelp.repeatCondition", template)
        self.assertIn('inputs is the current repeat memory', template)
        self.assertIn('class="node-run-condition" open', template)
        self.assertIn('class="node-default-output" open', template)
        self.assertIn("[hidden],", base)
        self.assertIn("if (['equals', 'not_equals'].includes(mode))", payload_body)
        self.assertIn("if (mode === 'ref')", payload_body)
        self.assertNotIn("compare_value:", payload_body[payload_body.index("const payload"):payload_body.index("if (['equals'")])

    def test_fallback_output_editor_shows_inferred_output_fields(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function nodeDefaultOutputSection")
        end = template.index("function conditionLengthControls", start)
        body = template[start:end]

        self.assertIn("fallbackOutputEditor(prefix, node)", body)
        self.assertIn("fallbackOutputFieldsForNode(node)", body)
        self.assertIn("outputFieldsForSpec(node)", body)
        self.assertIn("fallbackOutputRow(prefix, field, defaults[field], true)", body)
        self.assertIn("fieldHelp.fallbackOutput", body)
        self.assertIn("$input.solution", template)
        self.assertIn("$state.solution", template)
        self.assertIn("$node.solver.solution", template)
        self.assertIn("collectFallbackOutputEntries(prefix)", template)
        self.assertNotIn("Add fallback value", body)
        self.assertIn(".fallback-output-row", base)

    def test_preset_delete_buttons_use_delegated_data_handler(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_presets.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertNotIn('onclick="deleteAgent(', template)
        self.assertIn("data-delete-agent", template)
        self.assertIn("button.dataset.agentName", template)
        self.assertIn("button.dataset.deleteUrl", template)
        self.assertIn("cursor: pointer;", base[base.index(".button-action {"):])

    def test_editor_supports_ctrl_drag_multi_selection_and_group_copy(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertIn("id=\"selection-rectangle\"", template)
        self.assertIn("function startCanvasSelection", template)
        self.assertIn("function isMultiSelectEvent", template)
        self.assertIn("isMultiSelectEvent(event) && startCanvasSelection(event)", template)
        self.assertIn("state.canvas.scrollLeft", template)
        self.assertIn("state.canvas.scrollTop", template)
        self.assertIn("setCanvasMultiSelection(nodesInsideSelection(selectionDragBounds())", template)
        self.assertIn("op: 'copy_nodes'", template)
        paste_start = template.index("async function pasteCopiedGraphNode")
        paste_end = template.index("function pastePositionForNode", paste_start)
        paste_body = template[paste_start:paste_end]
        self.assertNotIn("displayLabel(node)} copy", paste_body)
        self.assertNotIn("label:", paste_body)
        self.assertIn(".selection-rectangle", base)

    def test_registry_poll_does_not_rerender_focused_prompt_editor(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function applyRegistryData")
        end = template.index("function applyEditorData", start)
        body = template[start:end]

        self.assertNotIn("renderNodeInspector()", body)
        self.assertIn("promptAutosaveTimers.size > 0", template)
        self.assertIn("hasActiveEditorControl()", template)
        self.assertIn("hasPendingWorkflowAutosave() || hasActiveEditorControl()", template)

    def test_prompt_output_fields_do_not_add_synthetic_text_socket(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function promptOutputFields")
        end = template.index("function hasPromptConfig", start)
        body = template[start:end]

        self.assertNotIn("fields.add('text')", body)
        self.assertIn("if (output.default_field) fields.add(output.default_field)", body)

    def test_dependency_edges_do_not_add_complete_output_socket(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function graphOutputFields")
        end = template.index("function orderSocketFields", start)
        body = template[start:end]

        self.assertIn("if (edge.edge_kind === 'dependency') continue;", body)

    def test_existing_workflow_inputs_only_edit_default_values(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function workflowInputsEditorHtml")
        body = template[start:template.index("function workflowBudgetEditorHtml", start)]
        start = template.index("function mappingEditor")
        mapping_body = template[start:template.index("function wiringHeader", start)]
        start = template.index("function wiringRow")
        row_body = template[start:template.index("function customWiringValueControl", start)]

        self.assertIn("displayFields: true", body)
        self.assertIn("allowAdd: true", body)
        self.assertIn("allowRemove: true", body)
        self.assertIn("lockedFields: workflowNames", body)
        self.assertIn("Add workflow input", body)
        self.assertIn("const allowRemove = options.allowRemove !== false;", mapping_body)
        self.assertIn("const displayFields = Boolean(options.displayFields);", mapping_body)
        self.assertIn("displayField: displayFields", mapping_body)
        self.assertIn("allowRemove", mapping_body)
        self.assertIn("const displayField = Boolean(rowOptions.displayField);", row_body)
        self.assertIn("rowOptions.allowRemove === false", row_body)
        self.assertIn(".wiring-rows", base)
        self.assertIn(".io-panel .wiring-field-display", base)

    def test_input_editor_add_row_uses_plain_name_field(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function parameterEditor")
        end = template.index("function parameterEntriesForSpec", start)
        editor_body = template[start:end]
        start = template.index("function addParameterRow")
        end = template.index("function removeParameterRow", start)
        add_body = template[start:end]

        self.assertIn('class="parameter-add-custom"', editor_body)
        self.assertNotIn("parameter-add-field", editor_body)
        self.assertNotIn("Custom input", editor_body)
        self.assertIn("custom?.value || ''", add_body)
        self.assertNotIn("select?.value", add_body)
        self.assertNotIn("function parameterFieldOptions", template)
        self.assertIn("inspectorSection('Inputs'", template)
        self.assertIn("Add input", template)
        self.assertIn("No fixed or workflow inputs set.", template)
        self.assertNotIn("inspectorSection('Parameters'", template)
        self.assertIn("grid-template-columns: minmax(160px, 1fr) auto;", base)

    def test_input_editor_keeps_literal_names_and_defaults_to_matching_workflow_input(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function parameterRow")
        end = template.index("function parameterModeForValue", start)
        row_body = template[start:end]
        start = template.index("function parameterValueControl")
        end = template.index("function defaultParameterValue", start)
        value_body = template[start:end]
        workflow_branch = value_body[
            value_body.index("if (mode === 'workflow')"):value_body.index("if (mode === 'expression')")
        ]
        start = template.index("function defaultParameterValue")
        end = template.index("function parameterEditorForPromptSpec", start)
        default_body = template[start:end]
        start = template.index("function collectParameterRowValue")
        end = template.index("function parameterContextFromRoot", start)
        collect_body = template[start:end]

        self.assertIn("inputFieldLabel(field)", row_body)
        self.assertNotIn("displayFieldName(field, context)", row_body)
        self.assertIn('type="hidden"', workflow_branch)
        self.assertNotIn("<select", workflow_branch)
        self.assertNotIn("workflowInputNames", workflow_branch)
        self.assertIn("return field ? `$input.${field}`", collect_body)
        self.assertNotIn("names.includes(field) ? field : names[0] || field", value_body)
        self.assertIn("return `$input.${field}`", default_body)
        self.assertIn('.parameter-row[data-parameter-mode="workflow"]', base)
        self.assertIn("grid-template-columns: minmax(100px, 0.45fr) minmax(180px, 1fr) auto;", base)

    def test_input_editor_hides_automatic_workflow_inputs_but_keeps_graph_wired_inputs(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function parameterEditor")
        end = template.index("function parameterEmptyState", start)
        body = template[start:end]
        start = template.index("function parameterRow")
        end = template.index("function parameterModeForValue", start)
        row_body = template[start:end]

        self.assertIn("parameterEntriesForSpec(spec, options.fields || [], context)", body)
        self.assertIn("const wired = wiredInputsForSpec(spec)", body)
        self.assertIn("fields", body)
        self.assertIn("isAutomaticWorkflowInputField(field, value)", body)
        self.assertIn("isAutomaticWorkflowInputField(field, wired[field])", body)
        self.assertIn("defaultParameterValue(field, context)", body)
        self.assertIn("containsNodeReference(value)", row_body)
        self.assertIn("parameter-row-connected", row_body)
        self.assertIn("sourceSummary(value, context)", row_body)
        self.assertIn(".parameter-row-connected", base)

    def test_prompt_sharing_dropdown_uses_representative_node_and_supports_untie(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function promptTiePanel")
        end = template.index("function renderPromptEditorBlock", start)
        body = template[start:end]

        self.assertIn("const label = displayLabel(group.nodes[0]);", body)
        self.assertIn("users.slice(0, 1)", body)
        self.assertNotIn("labels.join(', ')", body)
        self.assertIn("Untie this prompt", body)
        self.assertIn("function untiePromptFromSelected", template)
        self.assertIn("op: 'untie_component'", template)
        self.assertIn(".prompt-tie-actions-single", base)
        self.assertIn(".prompt-tie-actions select", base)
        self.assertIn("appearance: none;", base)
        self.assertIn("min-height: 38px;", base)
        self.assertIn(".prompt-tie-actions button", base)

    def test_empty_graph_ports_do_not_render_none_text(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function renderCanvasPorts")
        end = template.index("return fields.map", start)
        body = template[start:end]

        self.assertIn('class="canvas-port canvas-port-empty" aria-hidden="true"', body)
        self.assertNotIn(">None<", body)

    def test_graph_input_fields_hide_automatic_workflow_inputs(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function graphVisibleInputFieldsForNode")
        end = template.index("function containsNodeReference", start)
        body = template[start:end]

        self.assertIn("Object.keys(wired)", body)
        self.assertIn("Object.keys(node.inputs_schema || {})", body)
        self.assertIn("if (node.visual && Array.isArray(node.input_fields))", body)
        self.assertIn("!isAutomaticWorkflowInputField(field, wired[field])", body)
        self.assertIn("_specialGraphInputFieldsForNode(node)", body)
        self.assertIn("containsNodeReference(value)", body)
        self.assertNotIn("declaredInputFieldsForNode(node).filter", body)
        self.assertNotIn("detectedPromptVariables", body)
        self.assertNotIn("graphRequiredInputFieldsForNode", template)
        self.assertIn("if (field === 'workspace') return false;", template)

    def test_graph_ports_do_not_rename_solution_to_current_proof(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function portLabel")
        end = template.index("function startConnectionDrag", start)
        body = template[start:end]

        self.assertNotIn("solution: 'current proof'", body)
        self.assertNotIn("current proof carried through this step", body)

    def test_current_output_rows_autosave_renames_without_duplicate_labels(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        start = template.index("function currentPromptOutputsEditor")
        end = template.index("function configuredPromptOutputEntries", start)
        body = template[start:end]
        start = template.index("function scheduleCurrentOutputRename")
        end = template.index("async function removeCurrentOutput", start)
        rename_body = template[start:end]

        self.assertNotIn("current-output-kind", body)
        self.assertNotIn(">Rename<", body)
        self.assertIn("oninput=\"scheduleCurrentOutputRename", body)
        self.assertIn("data-current-output-original", body)
        self.assertIn("dataset.currentOutputOriginal", body)
        self.assertIn("scheduleSafeNodeSettingsAutosave", rename_body)
        self.assertIn("renderReport: false", rename_body)
        self.assertIn("grid-template-columns: minmax(150px, 1fr) auto;", base)

    def test_suppressed_wiring_click_cannot_select_node_without_opening_inspector(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("function startNodeDrag")
        end = template.index("function beginWiringGesture", start)
        body = template[start:end]
        guard_index = body.index("if (isCanvasNodeClickSuppressed())")
        select_index = body.index("setSelectedNode")
        start = template.index("function suppressUpcomingCanvasNodeClick")
        end = template.index("function isCanvasNodeClickSuppressed", start)
        suppress_body = template[start:end]

        self.assertLess(guard_index, select_index)
        self.assertIn("event.preventDefault()", body[guard_index:select_index])
        self.assertIn("durationMs = 320", suppress_body)
        self.assertNotIn("+ 1800", suppress_body)

    def test_condition_autosave_does_not_rerender_focused_editor(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_preset_editor.html").read_text()
        start = template.index("async function applyNodeCondition")
        end = template.index("async function applyNodeRunCondition", start)
        body = template[start:end]

        self.assertIn("op: 'update_node_condition'", body)
        self.assertIn("renderReport: false", body)


if __name__ == "__main__":
    unittest.main()
