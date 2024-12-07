<script setup lang="ts">
import { useHands } from '../composables/useHands'
import { useStorage } from '@vueuse/core'
import HandVisualizer from './HandVisualizer.vue'

const {
  handData,
  isConnected,
  error,
  palmPosition,
  handRotation,
  getLatency,
  getFps
} = useHands()

// Track frame rate and performance metrics
const fps = ref(0)
const latency = ref(0)
const isRecording = ref(false)
const recordedFrames = useStorage('debug-hand-recording', [])

// Update metrics every frame
useRafFn(() => {
  fps.value = getFps()
  latency.value = getLatency()
})

// Recording functionality
const toggleRecording = () => {
  isRecording.value = !isRecording.value
  if (isRecording.value) {
    recordedFrames.value = []
  }
}

watch(handData, (data) => {
  if (isRecording.value && data) {
    recordedFrames.value.push({
      timestamp: Date.now(),
      data: JSON.parse(JSON.stringify(data))
    })
  }
})
</script>

<template>
  <div class="fixed top-0 left-0 p-4 font-mono text-sm bg-black/80 text-green-400 m-4 rounded-lg">
    <div class="space-y-4">
      <!-- Connection Status & Basic Metrics -->
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full" :class="isConnected ? 'bg-green-500' : 'bg-red-500'" />
          <span>{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
        </div>
        <div class="flex gap-4">
          <span>FPS: {{ fps }}</span>
          <span>Latency: {{ latency }}ms</span>
        </div>
      </div>

      <!-- Error Display -->
      <div v-if="error" class="text-red-400">
        Error: {{ error }}
      </div>

      <!-- Hand Visualization -->
      <div v-if="handData?.raw_landmarks" class="border-t border-green-800 pt-4">
        <HandVisualizer :landmarks="handData.raw_landmarks" :confidence="handData.confidence" />
      </div>

      <!-- Recording Controls -->
      <div class="flex items-center gap-4 border-t border-green-800 pt-4">
        <button @click="toggleRecording" class="px-3 py-1 rounded bg-green-800 hover:bg-green-700">
          {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
        </button>
        <span v-if="isRecording" class="text-red-400">
          Recording: {{ recordedFrames.length }} frames
        </span>
      </div>

      <!-- Debug Data -->
      <div class="space-y-4 border-t border-green-800 pt-4">
        <div>
          <strong>Palm Position:</strong>
          <pre>{{ JSON.stringify(palmPosition, null, 2) }}</pre>
        </div>

        <div>
          <strong>Hand Rotation:</strong>
          <pre>{{ JSON.stringify(handRotation, null, 2) }}</pre>
        </div>

        <details>
          <summary class="cursor-pointer hover:text-green-300">Raw Landmarks</summary>
          <div class="max-h-60 overflow-y-auto mt-2">
            <pre>{{ JSON.stringify(handData?.raw_landmarks, null, 2) }}</pre>
          </div>
        </details>
      </div>
    </div>
  </div>
</template>