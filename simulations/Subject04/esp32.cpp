/*
 * ESP32 Stress Detection Edge Code
 * 
 * Protocol (PC -> ESP32):
 *   INIT:w1,w2,...,w15,b                     (send initial weights + bias once)
 *   FEAT:f1,f2,...,f15                       (send features for one window)
 *   LABEL:0 or LABEL:1                       (send true label)
 *   END_SHIFT                                (end of shift – ESP32 returns final weights)
 * 
 * ESP32 -> PC responses:
 *   PRED:0 or PRED:1                         (inference result)
 *   WEIGHTS:w1,w2,...,w15,b                  (final model after END_SHIFT)
 *   (also prints timing info to Serial for logging)
 */

#include <Arduino.h>
#include <esp_timer.h>
#include <math.h>

// ================ Configuration ================
#define NUM_FEATURES    15
#define LEARNING_RATE   0.1f
#define SGD_STEPS       10
#define GRAD_CLIP_NORM  1.0f
#define ZERO_THRESHOLD  1e-6f

// Global scaler (hardcoded – REPLACE WITH YOUR ACTUAL VALUES)
const float global_mean[NUM_FEATURES] = {
1.9039078338556659f, -0.00012552762937415474f, 0.047612894191690916f, 57.70142857142857f, 0.01741609813681121f, 
0.09995504243565458f, 0.4926881395212058f, 0.3978163661518593f, 0.052729284967041615f, 1.9043590277999731f, 
1.9029840518480965f, 1.8673555922415108f, 1.9413816041579204f, 0.0058135109626301f, 0.0006793266640687747f
};

const float global_std[NUM_FEATURES] = {
2.4991358278585993f, 0.0011865488545259478f, 0.08990818476423618f, 15.638943998485162f, 0.03732657074804848f, 
0.20502594594097667f, 0.1357451435876711f, 0.2083093688127013f, 0.09656270898323065f, 2.4982334561092423f, 
2.501645342877903f, 2.4572480909130174f, 2.544318824826686f, 0.028540018158512922f, 0.003639151052683106f
};

// ================ Global Variables ================
float weights[NUM_FEATURES];
float bias;
float features[NUM_FEATURES];     // will be overwritten with normalized values

// Metrics
uint64_t total_inference_us = 0;
uint64_t total_update_us = 0;
uint32_t inference_count = 0;

// ================ Helper Functions ================

// Parse a comma-separated line of floats into an array.
// Returns true on success, false on parse error.
bool parseFloats(const char* line, float* out, int count) {
  char buffer[256];
  strncpy(buffer, line, sizeof(buffer) - 1);
  buffer[sizeof(buffer) - 1] = '\0';
  
  char* token = strtok(buffer, ",");
  for (int i = 0; i < count; i++) {
    if (token == nullptr) return false;
    out[i] = atof(token);
    token = strtok(nullptr, ",");
  }
  return true;
}

// Compute dot product of two float arrays of length n
float dotProduct(const float* a, const float* b, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// Sigmoid function
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Normalize features using hardcoded global scaler
void normalizeFeatures() {
  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = (features[i] - global_mean[i]) / global_std[i];
  }
}

// Run inference on currently normalized features.
// Returns stress probability (0..1).
float runInference() {
  float score = dotProduct(weights, norm_features, NUM_FEATURES) + bias;
  return sigmoid(score);
}

// Perform aggressive SGD update using the current normalized features and true label.
bool sgdUpdate(float true_label, float current_prob) {
  float error = true_label - current_prob;
  if (fabs(error) < ZERO_THRESHOLD) return false; // no correction needed
  for (int step = 0; step < SGD_STEPS; step++) {
    // Update weights with gradient clipping
    for (int i = 0; i < NUM_FEATURES; i++) {
      float grad = LEARNING_RATE * error * features[i];
      if (grad > GRAD_CLIP_NORM) grad = GRAD_CLIP_NORM;
      if (grad < -GRAD_CLIP_NORM) grad = -GRAD_CLIP_NORM;
      weights[i] += grad;
    }
    // Update bias
    bias += LEARNING_RATE * error;
    // Recompute error for next iteration? In classic aggressive SGD, we use the same error.
    // If you want to use the updated model for the next step's error, you would recompute prob.
    // We'll keep it simple: use the same error (single evaluation of gradient at current point).
    // Alternatively, you can recompute error with the updated weights. The difference is small.
  }
  
  return true;
}

// Send final weights + bias back to PC
void sendFinalWeights() {
  Serial.print("WEIGHTS:");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.print(weights[i], 6);
    if (i < NUM_FEATURES - 1) Serial.print(",");
  }
  Serial.print(",");
  Serial.println(bias, 6);
}

// ================ Setup ================
void setup() {
  Serial.begin(115200);
  while (!Serial);   // wait for PC connection
  
  Serial.println("ESP32 ready");
  Serial.println("Waiting for INIT command...");
  
  // Wait for INIT command with initial weights
  while (true) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.startsWith("INIT:")) {
        const char* data = line.c_str() + 5;  // skip "INIT:"
        if (parseFloats(data, weights, NUM_FEATURES)) {
          // After weights, there should be a comma and bias
          char* comma = strrchr(data, ',');
          if (comma != nullptr) {
            bias = atof(comma + 1);
          } else {
            Serial.println("ERROR: missing bias in INIT");
            continue;
          }
          Serial.println("Model loaded from PC (simulated smart card).");
          break;
        } else {
          Serial.println("ERROR: invalid INIT format");
        }
      }
    }
    delay(10);
  }
  
  // Report RAM footprint (approximate)
  size_t ram_used = sizeof(weights) + sizeof(bias) + sizeof(global_mean) + sizeof(global_std) + sizeof(features) + sizeof(norm_features) + 256; // buffers overhead
  Serial.print("RAM footprint estimate: ");
  Serial.print(ram_used);
  Serial.println(" bytes");
  Serial.print("Model size (weights+bias): ");
  Serial.print(sizeof(weights) + sizeof(bias));
  Serial.println(" bytes");
}

// ================ Main Loop ================
void loop() {
  if (!Serial.available()) {
    delay(5);
    return;
  }
  
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  // -------------------------------------------------
  // 1. Receive FEAT line (features for one window)
  // -------------------------------------------------
  if (command.startsWith("FEAT:")) {
    const char* data = command.c_str() + 5;
    if (!parseFloats(data, features, NUM_FEATURES)) {
      Serial.println("ERROR: invalid FEAT format");
      return;
    }
    
    // Normalize features
    normalizeFeatures();
    
    // Measure inference time
    uint64_t infer_start = esp_timer_get_time();
    float prob = runInference();
    uint64_t infer_end = esp_timer_get_time();
    uint64_t infer_delta = infer_end - infer_start;
    total_inference_us += infer_delta;
    inference_count++;
    
    bool stress = (prob >= 0.5f);
    Serial.print("PRED:");
    Serial.println(stress ? 1 : 0);
    
    // Wait for LABEL command (we expect it next)
    // We'll stay in this call until we get the label or a timeout.
    uint32_t timeout = 5000; // 5 seconds
    uint32_t start_ms = millis();
    bool label_received = false;
    float true_label = 0;
    
    while (millis() - start_ms < timeout) {
      if (Serial.available()) {
        String label_cmd = Serial.readStringUntil('\n');
        label_cmd.trim();
        if (label_cmd.startsWith("LABEL:")) {
          true_label = (label_cmd.charAt(6) == '1') ? 1.0f : 0.0f;
          label_received = true;
          break;
        }
      }
      delay(5);
    }
    
    if (!label_received) {
      Serial.println("ERROR: label timeout");
      return;
    }
    
    // Measure update time
    uint64_t update_start = esp_timer_get_time();
    sgdUpdate(true_label, prob);
    uint64_t update_end = esp_timer_get_time();
    uint64_t update_delta = update_end - update_start;
    total_update_us += update_delta;
    
    // (Optional) print timing info to PC – can be used for logging
    Serial.printf("INFER_TIME:%llu us\n", infer_delta);
    Serial.printf("UPDATE_TIME:%llu us\n", update_delta);
  }
  
  // -------------------------------------------------
  // 2. End of shift
  // -------------------------------------------------
  else if (command == "END_SHIFT") {
    // Send final weights back
    sendFinalWeights();
    
    // Report average timing
    if (inference_count > 0) {
      Serial.print("AVG_INFERENCE_US:");
      Serial.println(total_inference_us / inference_count);
      Serial.print("AVG_UPDATE_US:");
      Serial.println(total_update_us / inference_count);
      Serial.print("TOTAL_INFERENCES:");
      Serial.println(inference_count);
    }
    
    // Optionally reset state for new shift (but not needed, as device will be power-cycled)
    // We'll just idle.
    while (true) {
      delay(1000);
    }
  }
  
  // -------------------------------------------------
  // 3. Ignore other commands
  // -------------------------------------------------
  else {
    // Unknown command – ignore
    Serial.print("Unknown command: ");
    Serial.println(command);
  }
}