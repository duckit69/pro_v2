#include <Arduino.h>
#include <math.h>

// Note: Using 15 features instead of 18 because 3 of the WESAD columns are metadata 
// (subject, window_id, binary_label) and not valid features.
#define NUM_FEATURES 15
#define BUFFER_SIZE 10

struct ReplayItem {
    float features[NUM_FEATURES];
    uint8_t label;
};

// Global State
float current_weights[NUM_FEATURES];
float current_bias;
float scaler_mean[NUM_FEATURES];
float scaler_std[NUM_FEATURES];

ReplayItem replay_buffer[BUFFER_SIZE];
int buffer_idx = 0;
int buffer_count = 0;
int update_count = 0;

float lr = 0.001;

// Metrics tracking
int true_positives = 0;
int false_positives = 0;
int true_negatives = 0;
int false_negatives = 0;
int total_predictions = 0;
int correct_predictions = 0;

// Variables to store incoming features
float temp_features[NUM_FEATURES];
bool features_ready = false;
int last_prediction = 0;

// ==========================================
// STUB IMPLEMENTATIONS
// ==========================================

void loadModelFromSmartCard(float* weights, float* bias, float* mean, float* std) {
    // In production: Read from actual smart card
    // For now: initialized with 0.0 or a hardcoded generic model
    *bias = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        weights[i] = 0.0; 
        mean[i] = 0.0;
        std[i] = 1.0;
    }
    Serial.println("SYS: Loaded generic model from Smart Card.");
}

void saveModelToSmartCard(float* weights, float bias) {
    // In production: Write updated weights back to smart card
    // For now: print the final weights out
    Serial.println("SYS: Saving personalized model to Smart Card...");
    Serial.print("SYS: Final Bias: "); Serial.println(bias, 6);
    Serial.println("SYS: Final Weights:");
    for (int i = 0; i < NUM_FEATURES; i++) {
        Serial.print(weights[i], 6);
        if (i < NUM_FEATURES - 1) Serial.print(", ");
    }
    Serial.println();
}

// ==========================================
// CORE LOGIC
// ==========================================

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; } 

    // Simulate "start of shift" card swipe
    loadModelFromSmartCard(current_weights, &current_bias, scaler_mean, scaler_std);
    Serial.println("SYS: ESP32 Edge Learning Initialized.");
}

float sigmoid(float x) {
    if (x > 50.0) return 1.0;
    if (x < -50.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

float predict_prob(float* features) {
    float z = current_bias;
    for (int i = 0; i < NUM_FEATURES; i++) {
        z += current_weights[i] * features[i];
    }
    return sigmoid(z);
}

void sgd_update(float* features, uint8_t label) {
    float pred = predict_prob(features);
    float error = label - pred;
    
    current_bias += lr * error;
    for (int i = 0; i < NUM_FEATURES; i++) {
        current_weights[i] += lr * error * features[i];
    }
}

void process_replay_buffer() {
    for (int i = 0; i < buffer_count; i++) {
        sgd_update(replay_buffer[i].features, replay_buffer[i].label);
    }
}

void add_to_buffer(float* features, uint8_t label) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        replay_buffer[buffer_idx].features[i] = features[i];
    }
    replay_buffer[buffer_idx].label = label;
    
    buffer_idx = (buffer_idx + 1) % BUFFER_SIZE;
    if (buffer_count < BUFFER_SIZE) {
        buffer_count++;
    }
}

void parse_features(String line) {
    int feature_index = 0;
    int start_pos = 0;
    int comma_pos = line.indexOf(',');
    
    while (comma_pos != -1 && feature_index < NUM_FEATURES - 1) {
        temp_features[feature_index++] = line.substring(start_pos, comma_pos).toFloat();
        start_pos = comma_pos + 1;
        comma_pos = line.indexOf(',', start_pos);
    }
    if (feature_index < NUM_FEATURES) {
        temp_features[feature_index] = line.substring(start_pos).toFloat();
    }
    features_ready = true;
    
    float prob = predict_prob(temp_features);
    last_prediction = (prob >= 0.5) ? 1 : 0;
    
    // Exact format required by prompt
    Serial.print("Pred:");
    Serial.println(last_prediction);
}

void parse_label(String line) {
    if (!features_ready) {
        Serial.println("ERR: Label received before features.");
        return;
    }
    
    // "label:1"
    int colon_pos = line.indexOf(':');
    if (colon_pos == -1) return;
    
    uint8_t true_label = line.substring(colon_pos + 1).toInt();
    
    total_predictions++;
    if (last_prediction == true_label) {
        correct_predictions++;
        if (true_label == 1) true_positives++;
        else true_negatives++;
    } else {
        if (true_label == 1) false_negatives++; 
        else false_positives++; 
    }
    
    // SGD Update + Replay Logic
    sgd_update(temp_features, true_label);
    add_to_buffer(temp_features, true_label);
    update_count++;
    
    if (update_count % 5 == 0) {
        process_replay_buffer();
    }
    
    features_ready = false;
    Serial.println("ACK");
}

void handle_end_shift() {
    saveModelToSmartCard(current_weights, current_bias);
    
    float accuracy = (total_predictions > 0) ? ((float)correct_predictions / total_predictions) : 0.0;
    
    Serial.println("--- END OF SHIFT METRICS ---");
    Serial.print("Accuracy: "); Serial.println(accuracy, 4);
    Serial.println("Confusion Matrix:");
    Serial.print("[["); Serial.print(true_negatives); Serial.print("  "); Serial.print(false_positives); Serial.println("]");
    Serial.print(" ["); Serial.print(false_negatives); Serial.print("  "); Serial.print(true_positives); Serial.println("]]");
    
    // Reset state for safety, though device is theoretically powered off
    total_predictions = 0;
    correct_predictions = 0;
    true_positives = 0;
    false_positives = 0;
    true_negatives = 0;
    false_negatives = 0;
    update_count = 0;
    buffer_count = 0;
    buffer_idx = 0;
}

void loop() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        
        if (line.length() == 0) return;
        
        if (line == "END_SHIFT") {
            handle_end_shift();
        } 
        else if (line.startsWith("label:")) {
            parse_label(line);
        } 
        else {
            parse_features(line);
        }
    }
}
