#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DIMENSION 64  // Dimension des vecteurs d'attention
#define HEADS 8       // Nombre de têtes d'attention
#define MAX_LEN 256   // Longueur maximale des séquences
#define MAX_ENTRIES 1000 // Nombre maximum d'exemples
#define LEARNING_RATE 0.001

// Fonction softmax
void softmax(double *arr, int len) {
    double max_val = arr[0];
    double sum = 0.0;
    
    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    for (int i = 0; i < len; i++) {
        arr[i] = exp(arr[i] - max_val);
        sum += arr[i];
    }
    for (int i = 0; i < len; i++) {
        arr[i] /= sum;
    }
}

// Produit scalaire
double dot_product(const double *a, const double *b, int len) {
    double result = 0.0;
    for (int i = 0; i < len; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Calcul de l'attention
void multi_head_attention(const double *query, const double *key, const double *value, double *output, int len) {
    double attention_scores[len];
    
    for (int i = 0; i < len; i++) {
        attention_scores[i] = dot_product(&query[i * len], &key[i * len], len);
    }
    softmax(attention_scores, len);
    
    for (int i = 0; i < len; i++) {
        output[i] = 0.0;
        for (int j = 0; j < len; j++) {
            output[i] += attention_scores[j] * value[j * len + i];
        }
    }
}

// Feedforward
void feedforward(const double *input, double *output, int len) {
    double hidden[len];
    for (int i = 0; i < len; i++) {
        hidden[i] = fmax(0, input[i]);
        output[i] = hidden[i];
    }
}

// Encodeur Transformer
void transformer_encoder(const double *input, double *output, int len) {
    double attention_output[len];
    multi_head_attention(input, input, input, attention_output, len);
    feedforward(attention_output, output, len);
}

// Fonction de perte MSE
double mean_squared_error(const double *predicted, const double *actual, int len) {
    double loss = 0.0;
    for (int i = 0; i < len; i++) {
        loss += pow(predicted[i] - actual[i], 2);
    }
    return loss / len;
}

// Entraînement du modèle
void train_model(double *input_data, double *target_data, int data_size, int len) {
    double output[len];
    for (int i = 0; i < data_size; i++) {
        transformer_encoder(input_data + i * len, output, len);
        double loss = mean_squared_error(output, target_data + i * len, len);
        printf("Loss: %f\n", loss);
    }
}

// Encodage du texte en vecteurs
void encode_text_to_vector(const char *text, double *vector, int len) {
    memset(vector, 0, len * sizeof(double));
    int text_len = strlen(text) < len ? strlen(text) : len;
    for (int i = 0; i < text_len; i++) {
        vector[i] = (double)text[i];
    }
}

// Chargement des données d'entraînement
int load_train_data(const char *filename, double *input_data, double *target_data, int len) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Erreur ouverture fichier");
        return 0;
    }
    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), file) && count < MAX_ENTRIES) {
        char *delimiter = strchr(line, '|');
        if (delimiter) {
            *delimiter = '\0';
            encode_text_to_vector(line, input_data + count * len, len);
            encode_text_to_vector(delimiter + 1, target_data + count * len, len);
            count++;
        }
    }
    fclose(file);
    return count;
}

int main() {
    double input_data[MAX_ENTRIES * MAX_LEN];
    double target_data[MAX_ENTRIES * MAX_LEN];
    int data_size = load_train_data("database.txt", input_data, target_data, MAX_LEN);
    printf("Données chargées : %d exemples\n", data_size);
    train_model(input_data, target_data, data_size, MAX_LEN);
    return 0;
}
