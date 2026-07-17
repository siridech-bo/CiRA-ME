<template>
  <v-container fluid class="pa-6" style="max-width: 1100px;">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <v-btn icon size="small" variant="text" class="mr-2" @click="cancel">
        <v-icon>mdi-arrow-left</v-icon>
      </v-btn>
      <div>
        <h1 class="text-h5 font-weight-bold">
          {{ isEdit ? 'Edit Watcher' : 'New Folder Watcher' }}
        </h1>
        <p class="text-body-2 text-medium-emphasis mb-0">
          Poll a folder, run each row through a ME-LAB model, write results to a CSV.
        </p>
      </div>
    </div>

    <!-- Tab strip. On a NEW watcher there's nothing to file-manage yet, so we
         only render the config tab. Once created (i.e. isEdit), all six tabs
         become available. -->
    <v-card :loading="loading" class="mb-4">
      <v-tabs
        v-model="activeTab"
        color="primary"
        density="compact"
        show-arrows
      >
        <v-tab value="config">
          <v-icon start size="small">mdi-cog-outline</v-icon>
          Config
        </v-tab>
        <template v-if="isEdit">
          <v-tab value="input">
            <v-icon start size="small">mdi-tray-arrow-down</v-icon>
            Input Files
            <v-chip
              v-if="filesData.input.total"
              size="x-small"
              class="ml-2"
              variant="tonal"
            >
              {{ filesData.input.total }}
            </v-chip>
          </v-tab>
          <v-tab value="output">
            <v-icon start size="small">mdi-tray-arrow-up</v-icon>
            Output Files
            <v-chip
              v-if="filesData.output.total"
              size="x-small"
              class="ml-2"
              variant="tonal"
            >
              {{ filesData.output.total }}
            </v-chip>
          </v-tab>
          <v-tab value="error">
            <v-icon start size="small">mdi-alert-circle-outline</v-icon>
            Errors
            <v-chip
              v-if="filesData.error.total"
              size="x-small"
              class="ml-2"
              :color="filesData.error.total > 0 ? 'error' : undefined"
              variant="flat"
            >
              {{ filesData.error.total }}
            </v-chip>
          </v-tab>
          <v-tab value="history">
            <v-icon start size="small">mdi-history</v-icon>
            History
          </v-tab>
          <v-tab value="advanced">
            <v-icon start size="small">mdi-wrench-outline</v-icon>
            Advanced
          </v-tab>
        </template>
      </v-tabs>
    </v-card>

    <v-window v-model="activeTab">
      <!-- ═══════════════════════════════════════════════════════════════════
           CONFIG TAB — original watcher-edit form. Untouched behaviour.
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="config">
        <v-card class="pa-6" :loading="loading">
          <v-form ref="formRef" @submit.prevent="save">
            <v-text-field
              v-model="form.name"
              label="Name"
              placeholder="e.g. Machine 001 monitor"
              variant="outlined"
              density="comfortable"
              :rules="[v => !!v || 'Name is required']"
              class="mb-4"
            />

            <v-select
              v-model="form.endpoint_id"
              :items="endpointOptions"
              item-title="label"
              item-value="value"
              label="Endpoint"
              placeholder="Pick a trained ME-LAB model"
              variant="outlined"
              density="comfortable"
              :rules="[v => !!v || 'Endpoint is required']"
              :loading="endpointsLoading"
              :disabled="isEdit"
              class="mb-4"
              :hint="isEdit ? 'Endpoint is fixed once created. Delete this watcher and create a new one to change models.' : 'Only active endpoints appear here'"
              persistent-hint
            />

            <v-text-field
              v-model="form.input_folder"
              label="Input Folder"
              :placeholder="defaultInputFolder"
              variant="outlined"
              density="comfortable"
              :rules="[v => !!v || 'Input folder is required']"
              class="mb-4"
              hint="Any path visible inside the backend container"
              persistent-hint
              @update:model-value="onInputFolderInput"
            />

            <v-text-field
              v-model="form.output_folder"
              label="Output Folder"
              :placeholder="defaultOutputFolder"
              variant="outlined"
              density="comfortable"
              :rules="[v => !!v || 'Output folder is required']"
              class="mb-4"
              @update:model-value="onOutputFolderInput"
            />

            <div class="d-flex align-center gap-4 mb-4" style="flex-wrap: wrap;">
              <v-text-field
                v-model.number="form.poll_interval_s"
                label="Poll interval (seconds)"
                type="number"
                :min="10"
                :max="3600"
                variant="outlined"
                density="comfortable"
                style="max-width: 220px;"
                :rules="[
                  v => (v !== null && v !== undefined && v !== '') || 'Required',
                  v => v >= 10 || 'Minimum 10',
                  v => v <= 3600 || 'Maximum 3600',
                ]"
              />
              <v-text-field
                v-model="form.file_glob"
                label="File glob"
                placeholder="*.txt"
                variant="outlined"
                density="comfortable"
                hint="e.g. *.csv or machine_*.txt"
                persistent-hint
                style="max-width: 260px;"
              />
            </div>

            <div class="mb-2 text-body-2 font-weight-medium">Header Mode</div>
            <v-radio-group
              v-model="form.header_mode"
              inline
              hide-details
              class="mb-6"
            >
              <v-radio label="Auto (detect from first row)" value="auto" />
              <v-radio label="Headered" value="headered" />
              <v-radio label="Headerless" value="headerless" />
            </v-radio-group>

            <!-- ── Log Watcher: parse mode + per-mode config ──────────────── -->
            <v-select
              v-model="form.parse_mode"
              :items="parseModeOptions"
              item-title="label"
              item-value="value"
              label="Parse mode"
              variant="outlined"
              density="comfortable"
              class="mb-2"
              :hint="parseModeHint"
              persistent-hint
            />
            <div class="mb-2 d-flex justify-end">
              <v-btn
                variant="text"
                size="x-small"
                color="grey"
                :prepend-icon="showAdvancedParseModes ? 'mdi-eye-off-outline' : 'mdi-cog-outline'"
                @click="toggleAdvancedParseModes"
              >
                {{ showAdvancedParseModes ? 'Hide advanced parse modes' : 'Show advanced parse modes' }}
              </v-btn>
            </div>
            <div class="mb-4" />

            <!-- key_value mode: just list column names.
                 The "Pick a sample file" button on the right is the primary
                 way most users should populate this — Don't Make Me Think:
                 they upload a log line, we tell them what the columns are.
                 Typing them in by hand is the fallback. -->
            <div v-if="form.parse_mode === 'key_value'" class="d-flex align-start gap-2 mb-4">
              <v-text-field
                v-model="form.parse_columns"
                label="Column names (comma-separated)"
                placeholder="temperature, vibration, pressure"
                variant="outlined"
                density="comfortable"
                hint="Or use ➜ 'Pick sample file' to auto-fill. Watcher looks for these key=value or key:value pairs on each line."
                persistent-hint
                :rules="[
                  v => form.parse_mode !== 'key_value' || (!!v && !!String(v).trim()) || 'List at least one column'
                ]"
                style="flex: 1;"
              />
              <input
                ref="autoDetectFileInput"
                type="file"
                accept=".log,.txt,.csv,.jsonl"
                style="display: none;"
                @change="onAutoDetectFilePicked"
              />
              <v-btn
                color="primary"
                variant="tonal"
                :loading="autoDetectFileLoading"
                prepend-icon="mdi-file-search-outline"
                title="Pick a sample log file — the watcher will read its first few lines, detect the columns, and fill this field for you."
                style="height: 56px; white-space: nowrap;"
                @click="pickAutoDetectFile"
              >
                Pick sample file
              </v-btn>
            </div>

            <!-- regex mode: template picker + textarea -->
            <template v-if="form.parse_mode === 'regex'">
              <v-alert
                type="info"
                variant="tonal"
                density="compact"
                class="mb-3"
                icon="mdi-lightbulb-outline"
              >
                If your log has <code>key=value</code> or <code>key:value</code> pairs (like
                <code>temperature=45.32 vibration=0.87</code>), use
                <strong>Key = Value pairs</strong> mode instead — no regex needed.
                Regex is only for unusual formats.
              </v-alert>
              <v-select
                v-model="regexTemplate"
                :items="regexTemplateOptions"
                item-title="label"
                item-value="value"
                label="Template"
                variant="outlined"
                density="comfortable"
                class="mb-3"
                hint="Pick a starting point, then edit as needed. Choosing a template overwrites the regex below."
                persistent-hint
                @update:model-value="onRegexTemplateChange"
              />

              <v-textarea
                v-model="form.parse_regex"
                label="Parse regex"
                placeholder="(?P<time>\S+)\s+temp=(?P<temperature>\d+\.\d+)\s+vib=(?P<vibration>\d+\.\d+)"
                variant="outlined"
                density="comfortable"
                rows="2"
                auto-grow
                class="mb-4"
                style="font-family: monospace;"
                hint="Python regex with named capture groups. Each match becomes a row."
                persistent-hint
                :rules="[
                  v => form.parse_mode !== 'regex' || (!!v && !!v.trim()) || 'Regex is required'
                ]"
                @update:model-value="onRegexManualEdit"
              />
            </template>

            <!-- ── Live sample preview panel ──────────────────────────────── -->
            <v-expansion-panels v-model="previewPanel" class="mb-4">
              <v-expansion-panel>
                <v-expansion-panel-title>
                  <v-icon size="small" class="mr-2">mdi-flask-outline</v-icon>
                  Try a sample from your log file
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <v-alert type="info" variant="tonal" density="compact" class="mb-3" icon="mdi-information-outline">
                    Paste <strong>actual log lines</strong> from a file — <em>not</em> your regex/config.
                    Example: <code>2026-07-10T08:00:00 INFO | temperature=45.32 vibration=0.87 pressure=47.52</code>
                  </v-alert>
                  <v-textarea
                    v-model="form.samplePreviewText"
                    :rows="3"
                    variant="outlined"
                    density="comfortable"
                    label="Paste 1-5 raw log lines here"
                    placeholder="2026-07-10T08:00:00 INFO | temperature=45.32 vibration=0.87 pressure=47.52"
                    hide-details
                    class="mb-3"
                    style="font-family: monospace;"
                  />
                  <div class="d-flex align-center gap-2 mb-2 flex-wrap">
                    <v-btn
                      size="small"
                      variant="tonal"
                      color="primary"
                      :loading="previewLoading"
                      :disabled="!form.samplePreviewText || !form.samplePreviewText.trim()"
                      @click="runPreview"
                    >
                      Test parse
                    </v-btn>
                    <v-btn
                      v-if="form.parse_mode === 'key_value' && !form.parse_columns.trim() && form.samplePreviewText.trim()"
                      size="small"
                      variant="tonal"
                      color="warning"
                      prepend-icon="mdi-lightning-bolt-outline"
                      :loading="detectLoading"
                      @click="detectColumnsFromSample"
                      title="Scan the sample for key=value patterns and fill the Column names field"
                    >
                      Auto-detect columns
                    </v-btn>
                    <span v-if="previewResult" class="text-caption text-medium-emphasis">
                      {{ previewResult.row_count }} row{{ previewResult.row_count === 1 ? '' : 's' }} parsed
                    </span>
                  </div>

                  <v-alert
                    v-if="previewError"
                    type="error"
                    variant="tonal"
                    density="compact"
                    class="mb-2"
                  >
                    <div class="d-flex align-center gap-2 mb-1">
                      <v-chip
                        v-if="previewError.error_code"
                        size="x-small"
                        color="error"
                        variant="flat"
                      >
                        {{ previewError.error_code }}
                      </v-chip>
                      <strong>{{ previewError.error }}</strong>
                    </div>
                    <div v-if="previewError.hint" class="text-caption">
                      {{ previewError.hint }}
                    </div>
                  </v-alert>

                  <template v-if="previewResult && !previewError">
                    <div v-if="previewResult.warnings && previewResult.warnings.length" class="mb-2">
                      <v-chip
                        v-for="(w, i) in previewResult.warnings"
                        :key="i"
                        size="x-small"
                        color="warning"
                        variant="tonal"
                        class="mr-1"
                      >
                        {{ w }}
                      </v-chip>
                    </div>
                    <v-table v-if="previewResult.columns.length" density="compact" class="preview-table">
                      <thead>
                        <tr>
                          <th
                            v-for="(c, i) in previewResult.columns"
                            :key="i"
                            class="text-left"
                          >
                            {{ c }}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, ri) in previewResult.rows.slice(0, 5)" :key="ri">
                          <td v-for="(cell, ci) in row" :key="ci">
                            {{ cell === null || cell === undefined ? '—' : cell }}
                          </td>
                        </tr>
                      </tbody>
                    </v-table>
                    <div
                      v-else
                      class="text-caption text-medium-emphasis pa-2"
                    >
                      No columns detected. Check the parse configuration above.
                    </div>
                  </template>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>

            <!-- ── Log Watcher: MQTT publish sink ─────────────────────────── -->
            <v-switch
              v-model="form.mqtt_enabled"
              label="Publish predictions to MQTT"
              color="primary"
              density="comfortable"
              hide-details
              class="mb-2"
            />
            <v-text-field
              v-if="form.mqtt_enabled"
              v-model="form.mqtt_topic"
              label="MQTT topic"
              placeholder="alerts/{name}"
              variant="outlined"
              density="comfortable"
              class="mb-4"
              hint="{name} is replaced with the watcher's slug at publish time"
              persistent-hint
              :rules="[
                v => !form.mqtt_enabled || (!!v && !!String(v).trim()) || 'Topic is required'
              ]"
            />

            <!-- ── Log Watcher: daily aggregated CSV sink ─────────────────── -->
            <v-switch
              v-model="form.daily_csv_enabled"
              label="Write daily aggregated CSV"
              color="primary"
              density="comfortable"
              hide-details
              class="mb-2"
              :hint="dailyCsvHint"
              persistent-hint
            />
            <div class="mb-6" />

            <div class="d-flex align-center justify-end gap-2">
              <v-btn variant="text" :disabled="saving" @click="cancel">Cancel</v-btn>
              <v-btn
                color="primary"
                variant="flat"
                :loading="saving"
                type="submit"
              >
                {{ isEdit ? 'Save Changes' : 'Create Watcher' }}
              </v-btn>
            </div>
          </v-form>
        </v-card>
      </v-window-item>

      <!-- ═══════════════════════════════════════════════════════════════════
           INPUT FILES TAB
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="input">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">
              <v-icon start size="small">mdi-tray-arrow-down</v-icon>
              Input queue
            </h3>
            <v-spacer />
            <v-btn
              size="small"
              variant="text"
              prepend-icon="mdi-refresh"
              :loading="filesLoading"
              @click="loadFiles(false)"
            >
              Refresh
            </v-btn>
          </div>

          <!-- Drag-drop upload zone -->
          <div
            class="upload-zone mb-4"
            :class="{ 'is-drag': isDragging, 'is-busy': uploading }"
            @dragover.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @drop.prevent="onDrop"
            @click="triggerFileInput"
          >
            <input
              ref="fileInput"
              type="file"
              multiple
              style="display: none;"
              @change="onFileInputChange"
            />
            <v-icon size="42" color="primary" class="mb-1">mdi-cloud-upload-outline</v-icon>
            <div class="text-body-2 font-weight-medium">
              Drop files here or click to browse
            </div>
            <div class="text-caption text-medium-emphasis">
              Up to 100 MB per file · matches the watcher's file glob
              <code>{{ form.file_glob || '*.txt' }}</code>
            </div>
          </div>

          <!-- Per-file upload progress -->
          <div v-if="uploads.length" class="mb-4">
            <div
              v-for="(u, i) in uploads"
              :key="i"
              class="d-flex align-center gap-3 mb-2"
            >
              <v-icon size="small">mdi-file-outline</v-icon>
              <span class="text-body-2 flex-grow-0" style="min-width: 160px;">{{ u.name }}</span>
              <v-progress-linear
                :model-value="u.progress"
                :color="u.error ? 'error' : (u.done ? 'success' : 'primary')"
                height="6"
                rounded
                class="flex-grow-1"
              />
              <span class="text-caption text-medium-emphasis" style="min-width: 90px; text-align: right;">
                <template v-if="u.error">{{ u.error }}</template>
                <template v-else-if="u.done">Done</template>
                <template v-else>{{ u.progress }}%</template>
              </span>
            </div>
          </div>

          <v-alert
            v-if="!filesLoading && filesData.input.files.length === 0"
            type="info"
            variant="tonal"
            density="compact"
            class="mb-2"
          >
            No files waiting in the input folder. Drop a file into the zone above to get started.
          </v-alert>

          <v-table v-else-if="filesData.input.files.length" density="compact" hover>
            <thead>
              <tr>
                <th></th>
                <th>Name</th>
                <th class="text-right">Size</th>
                <th>Arrived</th>
                <th class="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in filesData.input.files" :key="f.name" :class="{ 'file-wont-be-processed': f.matches_glob === false }">
                <td>
                  <v-icon size="small" :color="f.matches_glob === false ? 'warning' : undefined">
                    {{ f.matches_glob === false ? 'mdi-file-alert-outline' : 'mdi-file-outline' }}
                  </v-icon>
                </td>
                <td class="font-weight-medium">
                  {{ f.name }}
                  <v-chip
                    v-if="f.matches_glob === false"
                    size="x-small"
                    color="warning"
                    variant="tonal"
                    class="ml-2"
                    :title="`Watcher only processes files matching ${form.file_glob || '*.txt'} — this file will be ignored by the runtime.`"
                  >
                    <v-icon start size="10">mdi-alert-circle-outline</v-icon>
                    won't be processed
                  </v-chip>
                </td>
                <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
                <td class="text-right">
                  <v-btn
                    icon size="x-small" variant="text" color="info"
                    title="Preview"
                    @click="openPreview('input', f.name)"
                  >
                    <v-icon size="small">mdi-eye-outline</v-icon>
                  </v-btn>
                  <v-btn
                    icon size="x-small" variant="text" color="error"
                    title="Delete"
                    @click="deleteOne('input', f.name)"
                  >
                    <v-icon size="small">mdi-delete-outline</v-icon>
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-window-item>

      <!-- ═══════════════════════════════════════════════════════════════════
           OUTPUT FILES TAB
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="output">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">
              <v-icon start size="small">mdi-tray-arrow-up</v-icon>
              Output CSVs
            </h3>
            <v-spacer />
            <v-btn
              size="small"
              variant="text"
              prepend-icon="mdi-refresh"
              :loading="filesLoading"
              @click="loadFiles(false)"
            >
              Refresh
            </v-btn>
          </div>

          <v-alert type="info" variant="tonal" density="compact" class="mb-3" icon="mdi-information-outline">
            Output files are kept forever. Download individually or as a batch.
          </v-alert>

          <div class="mb-3">
            <v-btn
              color="primary"
              variant="tonal"
              size="small"
              prepend-icon="mdi-folder-zip-outline"
              :disabled="!filesData.output.files.length"
              :loading="zipping"
              @click="downloadOutputZip"
            >
              Download all as .zip
            </v-btn>
          </div>

          <v-alert
            v-if="!filesLoading && filesData.output.files.length === 0"
            type="info"
            variant="tonal"
            density="compact"
          >
            No output files yet — nothing has been processed successfully.
          </v-alert>

          <v-table v-else-if="filesData.output.files.length" density="compact" hover>
            <thead>
              <tr>
                <th></th>
                <th>Name</th>
                <th class="text-right">Size</th>
                <th>Produced</th>
                <th class="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in filesData.output.files" :key="f.name">
                <td><v-icon size="small" color="success">mdi-file-check-outline</v-icon></td>
                <td class="font-weight-medium">{{ f.name }}</td>
                <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
                <td class="text-right">
                  <v-btn
                    icon size="x-small" variant="text" color="info"
                    title="Preview"
                    @click="openPreview('output', f.name)"
                  >
                    <v-icon size="small">mdi-eye-outline</v-icon>
                  </v-btn>
                  <v-btn
                    icon size="x-small" variant="text" color="primary"
                    title="Download"
                    @click="downloadOne('output', f.name)"
                  >
                    <v-icon size="small">mdi-download-outline</v-icon>
                  </v-btn>
                  <v-btn
                    icon size="x-small" variant="text" color="error"
                    title="Delete"
                    @click="deleteOne('output', f.name)"
                  >
                    <v-icon size="small">mdi-delete-outline</v-icon>
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-window-item>

      <!-- ═══════════════════════════════════════════════════════════════════
           ERROR FILES TAB
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="error">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">
              <v-icon start size="small" color="error">mdi-alert-circle-outline</v-icon>
              Errored files
            </h3>
            <v-spacer />
            <v-btn
              size="small"
              variant="text"
              prepend-icon="mdi-refresh"
              :loading="filesLoading"
              @click="loadFiles(false)"
            >
              Refresh
            </v-btn>
          </div>

          <!-- Info banner: only show when there ARE errored files. Explains
               the Retry action for the rows below. Suppressed on the empty
               state so the page doesn't look like it's flagging a problem
               when everything is clean. -->
          <v-alert
            v-if="!filesLoading && filesData.error.files.length > 0"
            type="warning"
            variant="tonal"
            density="compact"
            class="mb-3"
            icon="mdi-lifebuoy"
          >
            Files here failed to process. <strong>Retry</strong> moves them back to input for another try.
          </v-alert>

          <v-alert
            v-if="!filesLoading && filesData.error.files.length === 0"
            type="success"
            variant="tonal"
            density="compact"
          >
            No errored files — everything's been processed cleanly.
          </v-alert>

          <v-table v-else-if="filesData.error.files.length" density="compact" hover>
            <thead>
              <tr>
                <th></th>
                <th>Name / reason</th>
                <th class="text-right">Size</th>
                <th>Failed</th>
                <th class="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in filesData.error.files" :key="f.name">
                <td><v-icon size="small" color="error">mdi-file-alert-outline</v-icon></td>
                <td class="font-weight-medium">
                  {{ f.name }}
                  <!-- Root-cause line captured when the runtime moved the file to
                       _error/ (see folder_watcher_service.py). Click Preview
                       for the full traceback. -->
                  <div
                    v-if="f.error_reason"
                    class="text-caption text-error"
                    style="font-weight: 400; margin-top: 2px;"
                    :title="f.error_reason"
                  >
                    <v-icon size="10" class="mr-1">mdi-alert-circle-outline</v-icon>
                    {{ f.error_reason }}
                  </div>
                </td>
                <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
                <td class="text-right">
                  <v-btn
                    icon size="x-small" variant="text" color="info"
                    title="Preview"
                    @click="openPreview('error', f.name)"
                  >
                    <v-icon size="small">mdi-eye-outline</v-icon>
                  </v-btn>
                  <v-btn
                    icon size="x-small" variant="text" color="warning"
                    title="Retry — move back to input"
                    :loading="retrying === f.name"
                    @click="retryOne(f.name)"
                  >
                    <v-icon size="small">mdi-restart</v-icon>
                  </v-btn>
                  <v-btn
                    icon size="x-small" variant="text" color="error"
                    title="Delete"
                    @click="deleteOne('error', f.name)"
                  >
                    <v-icon size="small">mdi-delete-outline</v-icon>
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-window-item>

      <!-- ═══════════════════════════════════════════════════════════════════
           HISTORY TAB
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="history">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">
              <v-icon start size="small">mdi-history</v-icon>
              Processing history
            </h3>
            <v-spacer />
            <v-btn
              size="small"
              variant="text"
              prepend-icon="mdi-refresh"
              :loading="historyLoading"
              @click="loadHistory"
            >
              Refresh
            </v-btn>
          </div>

          <v-alert type="info" variant="tonal" density="compact" class="mb-3" icon="mdi-information-outline">
            Each row shows an input file and the output it produced. Input files are kept in
            <code>_processed/</code> after successful runs.
          </v-alert>

          <v-alert
            v-if="!historyLoading && history.length === 0"
            type="info"
            variant="tonal"
            density="compact"
          >
            No processing runs yet. Drop a file into <strong>Input Files</strong> to get started.
          </v-alert>

          <v-table v-else-if="history.length" density="compact" hover>
            <thead>
              <tr>
                <th></th>
                <th>Input → Output</th>
                <th>Processed at</th>
                <th class="text-right">Output size</th>
                <th class="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(h, i) in history" :key="i">
                <td>
                  <v-icon v-if="h.status === 'success'" size="small" color="success">mdi-check-circle-outline</v-icon>
                  <v-icon v-else-if="h.status === 'output_only'" size="small" color="info">mdi-help-circle-outline</v-icon>
                  <v-icon v-else size="small" color="warning">mdi-alert-outline</v-icon>
                </td>
                <td>
                  <div class="d-flex align-center gap-2 flex-wrap">
                    <span v-if="h.input_name" class="font-weight-medium">{{ h.input_name }}</span>
                    <span v-else class="text-caption text-medium-emphasis">(no input)</span>
                    <v-icon size="x-small">mdi-arrow-right</v-icon>
                    <span v-if="h.output_name" class="font-weight-medium">{{ h.output_name }}</span>
                    <span v-else class="text-caption text-medium-emphasis">(no output)</span>
                  </div>
                </td>
                <td class="text-caption">{{ formatDate((h.processed_at || 0) * 1000) }}</td>
                <td class="text-right text-caption">
                  {{ h.output_size !== null && h.output_size !== undefined ? formatSize(h.output_size) : '—' }}
                </td>
                <td class="text-right">
                  <v-btn
                    v-if="h.output_name"
                    icon size="x-small" variant="text" color="primary"
                    title="Download output"
                    @click="downloadOne('output', h.output_name)"
                  >
                    <v-icon size="small">mdi-download-outline</v-icon>
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-window-item>

      <!-- ═══════════════════════════════════════════════════════════════════
           ADVANCED TAB — path introspection
      ═══════════════════════════════════════════════════════════════════ -->
      <v-window-item value="advanced">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">
              <v-icon start size="small">mdi-wrench-outline</v-icon>
              Advanced — direct paths & API URLs
            </h3>
            <v-spacer />
            <v-btn
              size="small"
              variant="text"
              prepend-icon="mdi-refresh"
              :loading="pathsLoading"
              @click="loadPaths"
            >
              Refresh
            </v-btn>
          </div>

          <v-alert type="info" variant="tonal" density="compact" class="mb-3" icon="mdi-information-outline">
            Use these paths for rsync, scp, or direct API automation. If you don't need this, you can ignore this tab.
          </v-alert>

          <v-alert
            v-if="paths && !paths.watcher_host_base_configured"
            type="warning"
            variant="tonal"
            density="compact"
            class="mb-3"
            icon="mdi-alert-outline"
          >
            <code>WATCHER_HOST_BASE_PATH</code> is not set on this deployment, so host paths
            can't be resolved. Set it on the backend container to enable them.
          </v-alert>

          <div v-for="kind in advancedKinds" :key="kind" class="mb-4">
            <div class="text-body-2 font-weight-bold mb-1 d-flex align-center">
              <v-icon start size="small">{{ advancedIcons[kind] }}</v-icon>
              {{ advancedLabels[kind] }}
            </div>
            <v-table v-if="paths" density="compact" class="paths-table">
              <tbody>
                <tr v-for="(row, idx) in advancedRowsFor(kind)" :key="idx">
                  <td class="text-caption" style="width: 140px;">{{ row.label }}</td>
                  <td>
                    <code v-if="row.value" class="text-caption">{{ row.value }}</code>
                    <span v-else class="text-caption text-medium-emphasis">not available</span>
                  </td>
                  <td class="text-right" style="width: 40px;">
                    <v-btn
                      v-if="row.value"
                      icon size="x-small" variant="text"
                      title="Copy to clipboard"
                      @click="copyToClipboard(row.value)"
                    >
                      <v-icon size="small">mdi-content-copy</v-icon>
                    </v-btn>
                  </td>
                </tr>
              </tbody>
            </v-table>
            <div v-else class="text-caption text-medium-emphasis">Loading…</div>
          </div>
        </v-card>
      </v-window-item>
    </v-window>

    <!-- ═════════════════════════════════════════════════════════════════════
         PREVIEW DIALOG — reused across input/output/error/processed tabs.
    ═════════════════════════════════════════════════════════════════════ -->
    <v-dialog v-model="showPreviewDialog" max-width="900" scrollable>
      <v-card v-if="previewFile">
        <v-card-title class="pt-4 pb-2 px-5">
          <div class="d-flex align-center">
            <v-icon class="mr-2">mdi-eye-outline</v-icon>
            <span class="text-truncate">{{ previewFile.name }}</span>
            <v-chip size="x-small" variant="tonal" class="ml-2">{{ previewFile.kind }}</v-chip>
            <v-spacer />
            <v-btn
              icon size="small" variant="text"
              :loading="previewFileLoading"
              @click="loadPreviewContent"
            >
              <v-icon>mdi-refresh</v-icon>
            </v-btn>
          </div>
        </v-card-title>
        <v-card-text style="max-height: 60vh;">
          <div
            v-if="previewFileData?.truncated"
            class="text-caption text-warning mb-2"
          >
            (truncated to first 200 KB — download for the full file)
          </div>
          <v-sheet
            v-if="previewFileData"
            color="grey-darken-4"
            class="pa-3"
            rounded
            style="max-height: 50vh; overflow: auto;"
          >
            <pre style="margin: 0; font-size: 11px; white-space: pre; font-family: monospace;">{{ previewFileData.content }}</pre>
          </v-sheet>
          <v-progress-linear v-else-if="previewFileLoading" indeterminate />
        </v-card-text>
        <v-card-actions class="px-5 pb-4">
          <v-btn
            variant="tonal"
            color="primary"
            prepend-icon="mdi-download-outline"
            @click="downloadOne(previewFile.kind, previewFile.name)"
          >
            Download
          </v-btn>
          <v-spacer />
          <v-btn variant="text" @click="showPreviewDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import api from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'

interface Endpoint {
  id: string
  name: string
  algorithm: string
  mode: string
  status: string
}

interface FileInfo {
  name: string
  size: number
  mtime: number
}

interface FolderData {
  files: FileInfo[]
  total: number
  folder: string
}

interface HistoryEntry {
  input_name: string | null
  archive_name: string | null
  output_name: string | null
  processed_at: number | null
  output_size: number | null
  status: 'success' | 'output_only' | 'archived_no_output'
}

interface PathEntry {
  container: string
  host: string | null
  api_upload?: string
  api_download_zip?: string
}

interface PathsResponse {
  input: PathEntry
  output: PathEntry
  processed: PathEntry
  error: PathEntry
  watcher_host_base_configured: boolean
}

type FileKind = 'input' | 'output' | 'error' | 'processed'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const notify = useNotificationStore()

const formRef = ref<any>(null)

const isEdit = computed(() => !!route.params.id)
const watcherId = computed(() =>
  route.params.id ? Number(route.params.id) : null
)

const loading = ref(false)
const saving = ref(false)
const endpointsLoading = ref(false)

const endpoints = ref<Endpoint[]>([])

const activeTab = ref<string>('config')

const form = ref({
  name: '',
  endpoint_id: '',
  input_folder: '',
  output_folder: '',
  poll_interval_s: 60,
  file_glob: '*.txt',
  header_mode: 'auto' as 'auto' | 'headered' | 'headerless',
  // New watchers default to key_value; existing watchers load their saved mode
  // in loadWatcher() so this default only matters for the "New" flow.
  parse_mode: 'key_value' as 'csv' | 'regex' | 'json' | 'key_value',
  parse_regex: '',
  parse_columns: '',
  mqtt_enabled: false,
  mqtt_topic: 'alerts/{name}',
  daily_csv_enabled: false,
  // Not persisted — just used by the "Try a sample line" panel
  samplePreviewText: '',
})

// Advanced parse modes are hidden by default. Regex is the only advanced
// mode today — kept out of the way of factory operators who won't write it.
const showAdvancedParseModes = ref(false)
const BASIC_PARSE_MODES = ['key_value', 'json', 'csv']
const ADVANCED_PARSE_MODES = ['regex']
const parseModeOptions = computed(() => {
  const base = [
    { value: 'key_value', label: 'Key = Value pairs (recommended)' },
    { value: 'json',      label: 'JSON — one object per line' },
    { value: 'csv',       label: 'CSV — headered rows' },
  ]
  if (showAdvancedParseModes.value) {
    base.push({ value: 'regex', label: 'Regex (named groups per line) — advanced' })
  }
  return base
})

function toggleAdvancedParseModes() {
  showAdvancedParseModes.value = !showAdvancedParseModes.value
  if (!showAdvancedParseModes.value && ADVANCED_PARSE_MODES.includes(form.value.parse_mode as any)) {
    form.value.parse_mode = 'key_value'
    form.value.parse_regex = ''
  }
}

const parseModeHints: Record<string, string> = {
  key_value: "Each line's key=value pairs are extracted by column name. Simplest for factory logs.",
  regex:     'Full regex power with named capture groups. Best when the format is unusual.',
  json:      'Each line must be a valid JSON object. Best when logs are already structured.',
  csv:       "Each file's first row is the header, subsequent rows are records.",
}
const parseModeHint = computed(() =>
  parseModeHints[form.value.parse_mode] || parseModeHints.key_value
)

// ── Regex templates ──────────────────────────────────────────────────────
const regexTemplate = ref<string>('')
const regexTemplateOptions = [
  { value: '',          label: 'Custom (write your own)' },
  { value: 'apache',    label: 'Apache-style access log' },
  { value: 'syslog',    label: 'Syslog line' },
  { value: 'csv_line',  label: 'CSV-like (comma-separated per line)' },
  { value: 'space_sep', label: 'Space-separated numbers' },
  { value: 'factory',   label: 'Factory sensor line (temperature / vibration / pressure)' },
]
const regexTemplates: Record<string, string> = {
  apache:    '^(?P<ip>\\S+)\\s+\\S+\\s+(?P<user>\\S+)\\s+\\[(?P<time>[^\\]]+)\\]\\s+"(?P<method>\\S+)\\s+(?P<path>\\S+)\\s+HTTP/[\\d.]+"\\s+(?P<status>\\d+)\\s+(?P<bytes>\\d+)',
  syslog:    '^(?P<time>\\S+\\s+\\S+\\s+\\S+)\\s+(?P<host>\\S+)\\s+(?P<process>\\S+?)(\\[(?P<pid>\\d+)\\])?:\\s+(?P<message>.*)$',
  csv_line:  '^(?P<col1>[^,]*),(?P<col2>[^,]*),(?P<col3>[^,]*)',
  space_sep: '^(?P<col1>\\S+)\\s+(?P<col2>\\S+)\\s+(?P<col3>\\S+)',
  factory:   'temperature=(?P<temperature>-?\\d+\\.?\\d*)\\s+vibration=(?P<vibration>-?\\d+\\.?\\d*)\\s+pressure=(?P<pressure>-?\\d+\\.?\\d*)',
}
let regexAutofilling = false
function onRegexTemplateChange(v: string) {
  if (!v) return
  regexAutofilling = true
  form.value.parse_regex = regexTemplates[v] || ''
  nextTick(() => { regexAutofilling = false })
}
function onRegexManualEdit() {
  if (regexAutofilling) return
  if (regexTemplate.value !== '') regexTemplate.value = ''
}

// ── Live sample preview (config tab) ─────────────────────────────────────
const previewPanel = ref<number | undefined>(undefined)
const previewLoading = ref(false)
const previewResult = ref<{
  columns: string[]
  rows: any[][]
  row_count: number
  skipped_lines: number
  warnings: string[]
} | null>(null)
const previewError = ref<{ error: string; error_code?: string; hint?: string } | null>(null)
const detectLoading = ref(false)

async function detectColumnsFromSample() {
  try {
    detectLoading.value = true
    const res = await api.post('/api/folder-watchers/detect-columns', {
      sample_content: form.value.samplePreviewText || '',
    })
    const cols: string[] = res.data?.columns || []
    if (cols.length === 0) {
      notify.showError('No key=value or key:value patterns found in the sample.')
      return
    }
    form.value.parse_columns = cols.join(', ')
    notify.showSuccess(
      `Detected ${cols.length} column${cols.length === 1 ? '' : 's'}: ${cols.join(', ')}. ` +
      `Trim the list if any look like non-sensor keys (pid, port, etc.).`
    )
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to detect columns from sample')
  } finally {
    detectLoading.value = false
  }
}

// One-click column auto-detection from a real log file.
// User clicks "Pick sample file" next to the columns field → OS file
// picker → we read the first ~20 lines client-side (enough for the
// detect-columns endpoint, no server upload of the whole file) → post
// to /detect-columns → fill parse_columns. Also stashes the sample
// into samplePreviewText so the "Try a sample" panel below carries it.
const autoDetectFileInput = ref<HTMLInputElement | null>(null)
const autoDetectFileLoading = ref(false)

function pickAutoDetectFile() {
  autoDetectFileInput.value?.click()
}

async function onAutoDetectFilePicked(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target?.files?.[0]
  if (!file) return
  autoDetectFileLoading.value = true
  try {
    // Read a small head of the file. Log lines are short; 8 KB is plenty
    // for the ~20 lines the detector needs and avoids loading a huge log.
    const HEAD_BYTES = 8 * 1024
    const blob = file.slice(0, HEAD_BYTES)
    const text = await blob.text()
    const lines = text.split(/\r?\n/).filter(l => l.trim()).slice(0, 20)
    if (lines.length === 0) {
      notify.showError(`${file.name} appears to be empty or unreadable as text.`)
      return
    }
    const sample = lines.join('\n')
    form.value.samplePreviewText = sample

    const res = await api.post('/api/folder-watchers/detect-columns', {
      sample_content: sample,
    })
    const cols: string[] = res.data?.columns || []
    if (cols.length === 0) {
      notify.showError(
        `Couldn't find any key=value or key:value patterns in ${file.name}. ` +
        `Peek at the file to confirm it looks like ` +
        `"temperature=45.32 vibration=0.87" and re-try.`
      )
      return
    }
    form.value.parse_columns = cols.join(', ')
    notify.showSuccess(
      `Detected ${cols.length} column${cols.length === 1 ? '' : 's'} from ${file.name}: ${cols.join(', ')}. ` +
      `Trim any non-sensor keys (pid, port, etc.).`
    )
  } catch (e: any) {
    notify.showError(e.response?.data?.error || e?.message || 'Failed to read sample file')
  } finally {
    autoDetectFileLoading.value = false
    // Clear the input so picking the SAME file again re-fires @change.
    if (target) target.value = ''
  }
}

async function runPreview() {
  previewError.value = null
  previewResult.value = null
  try {
    previewLoading.value = true
    if (
      form.value.parse_mode === 'key_value' &&
      !(form.value.parse_columns || '').trim() &&
      (form.value.samplePreviewText || '').trim()
    ) {
      try {
        const detectRes = await api.post('/api/folder-watchers/detect-columns', {
          sample_content: form.value.samplePreviewText,
        })
        const cols: string[] = detectRes.data?.columns || []
        if (cols.length > 0) {
          form.value.parse_columns = cols.join(', ')
          notify.showSuccess(
            `Auto-detected ${cols.length} column${cols.length === 1 ? '' : 's'}: ${cols.join(', ')}. ` +
            `Edit the Column names field to remove non-sensor keys.`
          )
        }
      } catch { /* fall through — preview endpoint will report the same error */ }
    }

    const payload: Record<string, any> = {
      parse_mode: form.value.parse_mode,
      sample_content: form.value.samplePreviewText || '',
    }
    if (form.value.parse_mode === 'regex') payload.parse_regex = form.value.parse_regex
    if (form.value.parse_mode === 'key_value') payload.parse_columns = form.value.parse_columns
    if (form.value.parse_mode === 'csv') payload.header_mode = form.value.header_mode
    const res = await api.post('/api/folder-watchers/preview-parse', payload)
    previewResult.value = res.data
  } catch (e: any) {
    const data = e.response?.data || {}
    previewError.value = {
      error: data.error || 'Preview failed',
      error_code: data.error_code,
      hint: data.hint,
    }
  } finally {
    previewLoading.value = false
  }
}

watch(
  () => [
    form.value.parse_mode,
    form.value.parse_regex,
    form.value.parse_columns,
    form.value.header_mode,
  ],
  () => {
    previewResult.value = null
    previewError.value = null
  }
)

const dailyCsvHint = computed(() =>
  form.value.daily_csv_enabled
    ? `Appends to shared/log_watcher/${nameSlug.value}/<YYYY-MM-DD>.csv on the server`
    : 'When enabled, every prediction is appended to a per-day aggregated CSV'
)

const hasUserEditedInputFolder = ref(false)
const hasUserEditedOutputFolder = ref(false)

const userSlug = computed(() => {
  const u = auth.user?.username || 'user'
  return String(u).replace(/[^a-zA-Z0-9_-]/g, '_')
})
const nameSlug = computed(() => {
  return (form.value.name || 'watcher')
    .toLowerCase()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'watcher'
})
const defaultInputFolder = computed(() =>
  `/app/watcher-data/${userSlug.value}/${nameSlug.value}/input`
)
const defaultOutputFolder = computed(() =>
  `/app/watcher-data/${userSlug.value}/${nameSlug.value}/output`
)

const endpointOptions = computed(() =>
  endpoints.value
    .filter(ep => ep.status === 'active')
    .map(ep => ({
      value: ep.id,
      label: `${ep.name} — ${ep.algorithm} (${ep.mode})`,
    }))
)

const loadEndpoints = async () => {
  try {
    endpointsLoading.value = true
    const res = await api.get('/api/melab/endpoints')
    endpoints.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load endpoints')
  } finally {
    endpointsLoading.value = false
  }
}

const loadWatcher = async () => {
  if (!watcherId.value) return
  try {
    loading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId.value}`)
    const w = res.data
    form.value = {
      name: w.name,
      endpoint_id: w.endpoint_id,
      input_folder: w.input_folder,
      output_folder: w.output_folder,
      poll_interval_s: w.poll_interval_s,
      file_glob: w.file_glob,
      header_mode: w.header_mode,
      parse_mode: (w.parse_mode as any) || 'csv',
      parse_regex: w.parse_regex || '',
      parse_columns: w.parse_columns || '',
      mqtt_enabled: !!w.mqtt_enabled,
      mqtt_topic: w.mqtt_topic || 'alerts/{name}',
      daily_csv_enabled: !!w.daily_csv_enabled,
      samplePreviewText: '',
    }
    if (ADVANCED_PARSE_MODES.includes(w.parse_mode)) {
      showAdvancedParseModes.value = true
    }
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Watcher not found')
    router.push({ name: 'folder-watcher-list' })
  } finally {
    loading.value = false
  }
}

const save = async () => {
  const { valid } = await formRef.value.validate()
  if (!valid) return
  try {
    saving.value = true
    const payload: Record<string, any> = { ...form.value }
    if (!payload.input_folder) payload.input_folder = defaultInputFolder.value
    if (!payload.output_folder) payload.output_folder = defaultOutputFolder.value
    if (payload.parse_mode !== 'regex') payload.parse_regex = null
    if (payload.parse_mode !== 'key_value') payload.parse_columns = null
    delete payload.samplePreviewText
    if (!payload.mqtt_enabled) payload.mqtt_topic = null

    if (isEdit.value) {
      delete payload.endpoint_id
      await api.patch(`/api/folder-watchers/${watcherId.value}`, payload)
      notify.showSuccess('Watcher updated')
    } else {
      await api.post('/api/folder-watchers/', payload)
      notify.showSuccess('Watcher created')
    }
    router.push({ name: 'folder-watcher-list' })
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to save watcher')
  } finally {
    saving.value = false
  }
}

const cancel = () => {
  router.push({ name: 'folder-watcher-list' })
}

let autofillingFolders = false

function setFolderAutofill(input: string, output: string) {
  autofillingFolders = true
  form.value.input_folder = input
  form.value.output_folder = output
  nextTick(() => { autofillingFolders = false })
}

function onInputFolderInput() {
  if (!autofillingFolders) hasUserEditedInputFolder.value = true
}
function onOutputFolderInput() {
  if (!autofillingFolders) hasUserEditedOutputFolder.value = true
}

watch(() => form.value.name, () => {
  if (isEdit.value) return
  const input = defaultInputFolder.value
  const output = defaultOutputFolder.value
  autofillingFolders = true
  if (!hasUserEditedInputFolder.value) form.value.input_folder = input
  if (!hasUserEditedOutputFolder.value) form.value.output_folder = output
  nextTick(() => { autofillingFolders = false })
})

// ═════════════════════════════════════════════════════════════════════════
// File manager state (input / output / error / history / advanced tabs)
// ═════════════════════════════════════════════════════════════════════════

const filesLoading = ref(false)
const filesData = ref<Record<FileKind, FolderData>>({
  input:     { files: [], total: 0, folder: '' },
  output:    { files: [], total: 0, folder: '' },
  error:     { files: [], total: 0, folder: '' },
  processed: { files: [], total: 0, folder: '' },
})

async function loadFiles(silent = true) {
  if (!isEdit.value || !watcherId.value) return
  try {
    if (!silent) filesLoading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId.value}/files`)
    filesData.value = res.data
  } catch (e: any) {
    if (!silent) notify.showError(e.response?.data?.error || 'Failed to load files')
  } finally {
    filesLoading.value = false
  }
}

// ── Upload ──────────────────────────────────────────────────────────────
interface UploadItem {
  name: string
  progress: number
  done: boolean
  error: string
}
const isDragging = ref(false)
const uploading = ref(false)
const uploads = ref<UploadItem[]>([])
const fileInput = ref<HTMLInputElement | null>(null)

function triggerFileInput() {
  fileInput.value?.click()
}

async function onFileInputChange(e: Event) {
  const target = e.target as HTMLInputElement
  const files = Array.from(target.files || [])
  if (files.length) await uploadFiles(files)
  // Reset the input so the same file can be reselected.
  if (target) target.value = ''
}

async function onDrop(e: DragEvent) {
  isDragging.value = false
  const files = Array.from(e.dataTransfer?.files || [])
  if (files.length) await uploadFiles(files)
}

async function uploadFiles(files: File[]) {
  if (!watcherId.value) return
  uploading.value = true
  // Track UI progress per-file. Uploads happen one at a time so the progress
  // bar reflects the actual byte transfer rather than a wall-of-parallel.
  const items: UploadItem[] = files.map(f => ({
    name: f.name, progress: 0, done: false, error: '',
  }))
  uploads.value = items

  try {
    for (let i = 0; i < files.length; i++) {
      const f = files[i]
      const fd = new FormData()
      fd.append('files', f, f.name)
      try {
        const resp = await api.post(
          `/api/folder-watchers/${watcherId.value}/upload`,
          fd,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (evt) => {
              if (evt.total) {
                items[i].progress = Math.round((evt.loaded / evt.total) * 100)
              }
            },
          }
        )
        // Response is 200 even for partial-success (some files rejected).
        // Check the response body's `errors[]` for THIS file specifically —
        // otherwise a silent server-side rejection looked like Done.
        const serverErrors = Array.isArray(resp.data?.errors) ? resp.data.errors : []
        const serverUploaded = Array.isArray(resp.data?.uploaded) ? resp.data.uploaded : []
        const errForThisFile = serverErrors.find((e: any) => e?.name === f.name)
        if (errForThisFile) {
          items[i].error = errForThisFile.reason || 'Upload rejected'
        } else if (serverUploaded.length === 0 && serverErrors.length > 0) {
          // Server returned all errors — pick the first as a fallback message.
          items[i].error = serverErrors[0]?.reason || 'Upload rejected'
        } else {
          items[i].progress = 100
          items[i].done = true
        }
      } catch (e: any) {
        items[i].error = e.response?.data?.error || 'Upload failed'
      }
    }
    const okCount = items.filter(u => u.done).length
    const badCount = items.filter(u => u.error).length
    if (okCount) notify.showSuccess(`Uploaded ${okCount} file${okCount === 1 ? '' : 's'}`)
    if (badCount) notify.showError(`${badCount} file${badCount === 1 ? '' : 's'} failed to upload`)
    await loadFiles(false)
  } finally {
    uploading.value = false
    // Clear the progress list after a moment so it doesn't linger between drops.
    setTimeout(() => {
      if (!uploading.value) uploads.value = []
    }, 2500)
  }
}

// ── Delete / retry / download ───────────────────────────────────────────
const retrying = ref<string | null>(null)

async function deleteOne(kind: FileKind, name: string) {
  if (!watcherId.value) return
  if (!confirm(`Delete "${name}"? This cannot be undone.`)) return
  try {
    await api.delete(
      `/api/folder-watchers/${watcherId.value}/files/${kind}/${encodeURIComponent(name)}`
    )
    notify.showSuccess(`Deleted ${name}`)
    await loadFiles(false)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Delete failed')
  }
}

async function retryOne(name: string) {
  if (!watcherId.value) return
  try {
    retrying.value = name
    await api.post(
      `/api/folder-watchers/${watcherId.value}/files/error/${encodeURIComponent(name)}/retry`
    )
    notify.showSuccess(`Moved ${name} back to input`)
    await loadFiles(false)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Retry failed')
  } finally {
    retrying.value = null
  }
}

async function downloadOne(kind: FileKind, name: string) {
  if (!watcherId.value) return
  try {
    const resp = await api.get(
      `/api/folder-watchers/${watcherId.value}/files/${kind}/${encodeURIComponent(name)}/download`,
      { responseType: 'blob' }
    )
    const blobUrl = window.URL.createObjectURL(new Blob([resp.data]))
    const a = document.createElement('a')
    a.href = blobUrl
    a.download = name
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(blobUrl)
    document.body.removeChild(a)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Download failed')
  }
}

const zipping = ref(false)
async function downloadOutputZip() {
  if (!watcherId.value) return
  try {
    zipping.value = true
    const resp = await api.get(
      `/api/folder-watchers/${watcherId.value}/files/output/zip`,
      { responseType: 'blob' }
    )
    // Prefer a filename we set on the frontend so the browser's Save dialog
    // proposes something sensible even if the response's Content-Disposition
    // is stripped by intermediate proxies.
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    const dlName = `${(form.value.name || 'watcher').replace(/[^A-Za-z0-9_-]+/g, '_')}_outputs_${ts}.zip`
    const blobUrl = window.URL.createObjectURL(new Blob([resp.data], { type: 'application/zip' }))
    const a = document.createElement('a')
    a.href = blobUrl
    a.download = dlName
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(blobUrl)
    document.body.removeChild(a)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Zip download failed')
  } finally {
    zipping.value = false
  }
}

// ── Preview dialog ──────────────────────────────────────────────────────
const showPreviewDialog = ref(false)
const previewFile = ref<{ kind: FileKind, name: string } | null>(null)
const previewFileData = ref<{ name: string, content: string, truncated: boolean, size: number } | null>(null)
const previewFileLoading = ref(false)

async function openPreview(kind: FileKind, name: string) {
  previewFile.value = { kind, name }
  previewFileData.value = null
  showPreviewDialog.value = true
  await loadPreviewContent()
}

async function loadPreviewContent() {
  if (!watcherId.value || !previewFile.value) return
  try {
    previewFileLoading.value = true
    const { kind, name } = previewFile.value
    const res = await api.get(
      `/api/folder-watchers/${watcherId.value}/files/${kind}/${encodeURIComponent(name)}`
    )
    previewFileData.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load preview')
    previewFileData.value = null
  } finally {
    previewFileLoading.value = false
  }
}

// ── History tab ─────────────────────────────────────────────────────────
const historyLoading = ref(false)
const history = ref<HistoryEntry[]>([])

async function loadHistory() {
  if (!watcherId.value) return
  try {
    historyLoading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId.value}/history`)
    history.value = res.data?.entries || []
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load history')
  } finally {
    historyLoading.value = false
  }
}

// ── Advanced tab: path introspection ────────────────────────────────────
const pathsLoading = ref(false)
const paths = ref<PathsResponse | null>(null)
const advancedKinds: FileKind[] = ['input', 'output', 'processed', 'error']
const advancedLabels: Record<FileKind, string> = {
  input:     'Input folder',
  output:    'Output folder',
  processed: 'Processed archive',
  error:     'Error folder',
}
const advancedIcons: Record<FileKind, string> = {
  input:     'mdi-tray-arrow-down',
  output:    'mdi-tray-arrow-up',
  processed: 'mdi-archive-outline',
  error:     'mdi-alert-circle-outline',
}

interface AdvancedRow { label: string, value: string }
function advancedRowsFor(kind: FileKind): AdvancedRow[] {
  if (!paths.value) return []
  const entry = paths.value[kind]
  const rows: AdvancedRow[] = [
    { label: 'Container path', value: entry.container },
    { label: 'Host path',      value: entry.host || '' },
  ]
  if (entry.api_upload) rows.push({ label: 'API upload URL', value: entry.api_upload })
  if (entry.api_download_zip) rows.push({ label: 'API zip URL', value: entry.api_download_zip })
  return rows
}

async function loadPaths() {
  if (!watcherId.value) return
  try {
    pathsLoading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId.value}/paths`)
    paths.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load paths')
  } finally {
    pathsLoading.value = false
  }
}

async function copyToClipboard(text: string) {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
    } else {
      // Fallback for insecure contexts (http on LAN) — Clipboard API is https-only in most browsers.
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    notify.showSuccess('Copied to clipboard')
  } catch {
    notify.showError('Could not copy. Copy manually: ' + text)
  }
}

// ── Tab-driven lazy loading ─────────────────────────────────────────────
// Re-fetch files any time the user pops into a file-managing tab, so counts
// and rows stay fresh. History/paths are lighter and only fetched on demand.
watch(activeTab, (v) => {
  if (!isEdit.value) return
  if (v === 'input' || v === 'output' || v === 'error') {
    loadFiles(false)
  } else if (v === 'history') {
    loadHistory()
  } else if (v === 'advanced') {
    loadPaths()
  }
})

// ── Formatting ──────────────────────────────────────────────────────────
function formatSize(bytes: number): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
function formatDate(dt?: string | number | null): string {
  if (dt === null || dt === undefined || dt === '') return '—'
  try {
    const d = new Date(dt as any)
    if (isNaN(d.getTime())) return String(dt)
    return d.toLocaleString()
  } catch { return String(dt) }
}

onMounted(async () => {
  await loadEndpoints()
  if (isEdit.value) {
    await loadWatcher()
    hasUserEditedInputFolder.value = true
    hasUserEditedOutputFolder.value = true
    // Preload the counts silently so the tab-strip chips are populated on
    // first paint. Individual tabs re-load their data on click.
    loadFiles(true)
  } else {
    setFolderAutofill(defaultInputFolder.value, defaultOutputFolder.value)
  }
})
</script>

<style scoped>
.preview-table {
  border: 1px solid rgb(var(--v-theme-outline-variant, 224, 224, 224));
  border-radius: 4px;
  max-height: 240px;
  overflow-y: auto;
}

/* Drag-drop upload zone. Fades to primary tint while a dragover is active. */
.upload-zone {
  border: 2px dashed rgba(var(--v-theme-on-surface), 0.24);
  border-radius: 8px;
  padding: 24px 16px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
}
.upload-zone:hover {
  border-color: rgb(var(--v-theme-primary));
  background-color: rgba(var(--v-theme-primary), 0.04);
}
.upload-zone.is-drag {
  border-color: rgb(var(--v-theme-primary));
  background-color: rgba(var(--v-theme-primary), 0.08);
}
.upload-zone.is-busy {
  pointer-events: none;
  opacity: 0.6;
}

.paths-table code {
  word-break: break-all;
  white-space: normal;
}
</style>
