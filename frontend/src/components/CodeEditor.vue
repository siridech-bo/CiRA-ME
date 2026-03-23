<template>
  <div ref="editorContainer" class="code-editor" :style="{ height: height || '400px' }"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { python } from '@codemirror/lang-python'
import { oneDark } from '@codemirror/theme-one-dark'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { bracketMatching, foldGutter, indentOnInput, syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language'
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete'
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search'

const props = defineProps<{
  modelValue: string
  height?: string
  readonly?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

const editorContainer = ref<HTMLElement | null>(null)
let view: EditorView | null = null

function createEditor(doc: string) {
  if (!editorContainer.value) return

  // Destroy previous instance if exists
  if (view) {
    view.destroy()
    view = null
  }

  const extensions = [
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    history(),
    foldGutter(),
    indentOnInput(),
    bracketMatching(),
    closeBrackets(),
    highlightSelectionMatches(),
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    python(),
    oneDark,
    keymap.of([
      ...defaultKeymap,
      ...historyKeymap,
      ...closeBracketsKeymap,
      ...searchKeymap,
      indentWithTab,
    ]),
    EditorView.updateListener.of((update) => {
      if (update.docChanged) {
        emit('update:modelValue', update.state.doc.toString())
      }
    }),
    EditorView.theme({
      '&': { fontSize: '13px' },
      '&.cm-editor': { height: '100%' },
      '.cm-scroller': { overflow: 'auto' },
      '.cm-content': { fontFamily: '"Fira Code", "Cascadia Code", "JetBrains Mono", monospace' },
      '.cm-gutters': { minWidth: '40px' },
    }),
  ]

  if (props.readonly) {
    extensions.push(EditorState.readOnly.of(true))
  }

  const state = EditorState.create({ doc, extensions })
  view = new EditorView({ state, parent: editorContainer.value })
}

onMounted(() => {
  createEditor(props.modelValue || '')
})

watch(() => props.modelValue, (newVal) => {
  if (!view) {
    // Editor not yet created, create it now
    nextTick(() => createEditor(newVal || ''))
    return
  }
  const current = view.state.doc.toString()
  if (current !== newVal) {
    view.dispatch({
      changes: { from: 0, to: current.length, insert: newVal || '' }
    })
  }
})

onBeforeUnmount(() => {
  view?.destroy()
  view = null
})
</script>

<style scoped>
.code-editor {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;
  overflow: hidden;
}
</style>
