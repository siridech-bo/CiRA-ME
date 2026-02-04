<template>
  <div class="logo-full" :class="{ 'logo-full--compact': compact }">
    <!-- Logo Icon -->
    <svg
      class="logo-icon"
      :width="compact ? 32 : 40"
      :height="compact ? 32 : 40"
      viewBox="0 0 48 48"
    >
      <!-- Background -->
      <rect x="4" y="4" width="40" height="40" rx="8" fill="currentColor" class="logo-bg"/>

      <!-- Wave pattern -->
      <path
        class="logo-wave"
        d="M 8 24 Q 16 16, 24 24 T 40 24"
        stroke="url(#logoGradient)"
        stroke-width="3"
        fill="none"
        stroke-linecap="round"
      />

      <!-- Gradient -->
      <defs>
        <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#6366F1"/>
          <stop offset="100%" style="stop-color:#22D3EE"/>
        </linearGradient>
      </defs>

      <!-- Data points -->
      <circle cx="12" cy="20" r="2.5" fill="#6366F1"/>
      <circle cx="24" cy="24" r="3" fill="#A855F7"/>
      <circle cx="36" cy="20" r="2.5" fill="#22D3EE"/>

      <!-- Edge indicator -->
      <line x1="24" y1="32" x2="24" y2="40" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>
      <rect x="20" y="38" width="8" height="3" rx="1" fill="#22D3EE"/>
    </svg>

    <!-- Text -->
    <div v-if="!iconOnly" class="logo-text">
      <span class="logo-text-cira">CiRA</span>
      <span class="logo-text-me">ME</span>
      <span v-if="showTagline && !compact" class="logo-tagline">Machine Intelligence for Edge</span>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  compact?: boolean
  iconOnly?: boolean
  showTagline?: boolean
}>()
</script>

<style scoped lang="scss">
.logo-full {
  display: flex;
  align-items: center;
  gap: 12px;

  &--compact {
    gap: 8px;
  }
}

.logo-icon {
  flex-shrink: 0;
}

.logo-bg {
  fill: rgba(30, 41, 59, 1);
}

.logo-wave {
  animation: waveFlow 3s ease-in-out infinite;
}

@keyframes waveFlow {
  0%, 100% {
    d: path('M 8 24 Q 16 16, 24 24 T 40 24');
  }
  50% {
    d: path('M 8 24 Q 16 32, 24 24 T 40 24');
  }
}

.logo-text {
  display: flex;
  flex-direction: column;
  line-height: 1.1;
}

.logo-text-cira {
  font-size: 1.25rem;
  font-weight: 700;
  color: rgb(var(--v-theme-on-surface));
  letter-spacing: 0.5px;
}

.logo-text-me {
  font-size: 1.25rem;
  font-weight: 300;
  background: linear-gradient(90deg, #6366F1, #22D3EE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 2px;
}

.logo-tagline {
  font-size: 0.65rem;
  font-weight: 400;
  color: rgb(var(--v-theme-on-surface-variant));
  margin-top: 2px;
  letter-spacing: 0.5px;
}

.logo-full--compact {
  .logo-text-cira,
  .logo-text-me {
    font-size: 1rem;
  }
}
</style>
