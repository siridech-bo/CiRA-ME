<template>
  <v-stepper
    :model-value="currentStepIndex"
    alt-labels
    class="pipeline-stepper"
    elevation="0"
    bg-color="transparent"
  >
    <v-stepper-header>
      <template v-for="(step, index) in visibleSteps" :key="step.value">
        <v-stepper-item
          :value="index + 1"
          :title="step.title"
          :subtitle="isStepSkipped(step.value) ? '(DL Mode)' : undefined"
          :icon="step.icon"
          :complete="isStepComplete(step.value)"
          :color="getStepColor(step.value)"
          :editable="isStepEditable(step.value)"
          :class="{ 'step-skipped': isStepSkipped(step.value) }"
          @click="navigateToStep(step)"
        />
        <v-divider v-if="index < visibleSteps.length - 1" />
      </template>
    </v-stepper-header>
  </v-stepper>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'

const props = defineProps<{
  currentStep: string
}>()

const router = useRouter()
const pipelineStore = usePipelineStore()

const allSteps = [
  { value: 'data', title: 'Data Source', icon: 'mdi-database', route: 'pipeline-data' },
  { value: 'windowing', title: 'Windowing', icon: 'mdi-tune-vertical', route: 'pipeline-windowing' },
  { value: 'features', title: 'Features', icon: 'mdi-auto-fix', route: 'pipeline-features' },
  { value: 'training', title: 'Training', icon: 'mdi-brain', route: 'pipeline-training' },
  { value: 'deploy', title: 'Deploy', icon: 'mdi-rocket-launch', route: 'pipeline-deploy' }
]

// In DL mode, show all steps but mark features as skipped
const visibleSteps = computed(() => allSteps)

const currentStepIndex = computed(() => {
  const index = visibleSteps.value.findIndex(s => s.value === props.currentStep)
  return index >= 0 ? index + 1 : 1
})

function isStepComplete(stepValue: string) {
  const status = pipelineStore.stepStatus
  const stepStatus = status[stepValue as keyof typeof status]
  return stepStatus === 'complete' || stepStatus === 'skipped'
}

function isStepSkipped(stepValue: string) {
  const status = pipelineStore.stepStatus
  return status[stepValue as keyof typeof status] === 'skipped'
}

function isStepEditable(stepValue: string) {
  const status = pipelineStore.stepStatus
  const stepStatus = status[stepValue as keyof typeof status]
  return stepStatus !== 'disabled' && stepStatus !== 'skipped'
}

function getStepColor(stepValue: string) {
  if (isStepSkipped(stepValue)) return 'grey'
  if (stepValue === props.currentStep) return 'primary'
  if (isStepComplete(stepValue)) return 'success'
  return undefined
}

function navigateToStep(step: typeof allSteps[0]) {
  if (isStepEditable(step.value)) {
    router.push({ name: step.route })
  }
}
</script>

<style scoped lang="scss">
.pipeline-stepper {
  :deep(.v-stepper-item) {
    cursor: pointer;

    &.v-stepper-item--disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }
  }

  .step-skipped {
    opacity: 0.6;
    cursor: not-allowed;

    :deep(.v-stepper-item__avatar) {
      text-decoration: line-through;
    }
  }
}
</style>
