<script setup lang="ts">
import { ref, watch } from 'vue'
import { useHands } from '../composables/useHands'

const { handData, isConnected, error, palmPosition, handRotation } = useHands()

// Track frame rate
const fps = ref(0)
let frameCount = 0
let lastTime = performance.now()

watch(handData, () => {
  frameCount++
  const currentTime = performance.now()
  const elapsed = currentTime - lastTime

  if (elapsed >= 1000) {
    fps.value = Math.round((frameCount * 1000) / elapsed)
    frameCount = 0
    lastTime = currentTime
  }
})
</script>

<template>
  <div class="fixed top-0 left-0 p-4 font-mono text-sm bg-black/80 text-green-400 m-4 rounded-lg">
    <div class="space-y-2">
      <div class="flex items-center gap-2">
        <div class="w-2 h-2 rounded-full" :class="isConnected ? 'bg-green-500' : 'bg-red-500'">
        </div>
        <span>{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
        <span class="ml-4">FPS: {{ fps }}</span>
      </div>

      <div v-if="error" class="text-red-400">
        Error: {{ error }}
      </div>

      <template v-if="handData">
        <div class="border-t border-green-800 pt-2">
          <strong>Palm Center:</strong>
          <pre>{{ JSON.stringify(palmPosition, null, 2) }}</pre>
        </div>

        <div class="border-t border-green-800 pt-2">
          <strong>Rotation:</strong>
          <pre>{{ JSON.stringify(handRotation, null, 2) }}</pre>
        </div>

        <div class="border-t border-green-800 pt-2">
          <strong>Raw Landmarks:</strong>
          <div class="max-h-60 overflow-y-auto">
            <pre>{{ JSON.stringify(handData.raw_landmarks, null, 2) }}</pre>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>