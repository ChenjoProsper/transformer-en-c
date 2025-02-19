#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DIMENSION 24
#define MAX_ENTRIES 1000
#define DATABASE_FILE "database.txt"

// Fonction softmax sécurisée
void softmax(double *arr, int len) {
    double max_val = arr[0], sum = 0.0;
    for (int i = 1; i < len; i++) if (arr[i] > max_val) max_val = arr[i];
    for (int i = 0; i < len; i++) {
        arr[i] = exp(arr[i] - max_val);
        sum += arr[i];
    }
    for (int i = 0; i < len; i++) arr[i] /= sum;
}

// Produit scalaire
double dot_product(const double *a, const double *b, int len) {
    double result = 0.0;
    for (int i = 0; i < len; i++) result += a[i] * b[i];
    return result;
}

// Calcul de la similarité cosinus
double cosine_similarity(const double *a, const double *b, int len) {
    double dot_prod = dot_product(a, b, len);
    double norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < len; i++) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot_prod / (sqrt(norm_a) * sqrt(norm_b));
}

// Encodage en vecteur
void encode_text_to_vector(const char *text, double *vector, int len) {
    memset(vector, 0, len * sizeof(double));
    int text_len = strlen(text) < len ? strlen(text) : len;
    for (int i = 0; i < text_len; i++) vector[i] = (double)text[i];
}

// Chargement des données
int load_train_data(double *K, double *V, char responses[MAX_ENTRIES][DIMENSION], int len) {
    FILE *file = fopen(DATABASE_FILE, "r");
    if (!file) return 0;
    char line[512]; int count = 0;
    while (fgets(line, sizeof(line), file) && count < MAX_ENTRIES) {
        char *delimiter = strchr(line, '|');
        if (delimiter) {
            *delimiter = '\0';
            encode_text_to_vector(line, K + count * len, len);
            encode_text_to_vector(delimiter + 1, V + count * len, len);
            strncpy(responses[count], delimiter + 1, DIMENSION);
            responses[count][DIMENSION - 1] = '\0';
            count++;
        }
    }
    fclose(file);
    return count;
}

// Trouver la meilleure correspondance avec l'attention (similarité cosinus)
void find_best_match(const double *Q, const double *K, const double *V, char responses[MAX_ENTRIES][DIMENSION], int num_samples, int len) {
    double best_score = -1.0;
    int best_index = 0;
    for (int i = 0; i < num_samples; i++) {
        double score = cosine_similarity(Q, &K[i * len], len);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    printf("Réponse : %s\n", responses[best_index]);
}

// Test du modèle
void test_model(const char *question, const double *K, const double *V, char responses[MAX_ENTRIES][DIMENSION], int num_samples) {
    double Q[DIMENSION] = {0};
    encode_text_to_vector(question, Q, DIMENSION);
    find_best_match(Q, K, V, responses, num_samples, DIMENSION);
}

void chat(double *K, double *V, char responses[MAX_ENTRIES][DIMENSION], int num_samples) {
    char input[256];
    while (1) {
        printf("\nVous : ");
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = 0;
        if (strcmp(input, "exit") == 0) break;
        test_model(input, K, V, responses, num_samples);
    }
}

int main() {
    double K[MAX_ENTRIES * DIMENSION] = {0};
    double V[MAX_ENTRIES * DIMENSION] = {0};
    char responses[MAX_ENTRIES][DIMENSION];
    int num_samples = load_train_data(K, V, responses, DIMENSION);
    printf("%d entrainements chargés\n", num_samples);
    chat(K, V, responses, num_samples);
    return 0;
}
