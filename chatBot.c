#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DIMENSION 24  // Nouvelle dimension des vecteurs d'attention
#define HEADS 8       // Nombre de têtes d'attention
#define MAX_LEN 256   // Longueur maximale des séquences
#define MAX_ENTRIES 1000 // Nombre maximum d'exemples
#define LEARNING_RATE 0.001
#define WEIGHTS_FILE "weights.bin"
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

// Calcul de l'attention
void multi_head_attention(const double *query, const double *key, const double *value, double *output, int len) {
    double attention_scores[len];
    for (int i = 0; i < len; i++) attention_scores[i] = dot_product(query, &key[i * len], len);
    softmax(attention_scores, len);
    for (int i = 0; i < len; i++) {
        output[i] = 0.0;
        for (int j = 0; j < len; j++) {
            output[i] += attention_scores[j] * value[j * len + i];
        }
    }
}

// Encodage du texte en vecteurs
void encode_text_to_vector(const char *text, double *vector, int len) {
    memset(vector, 0, len * sizeof(double));
    int text_len = strlen(text) < len ? strlen(text) : len;
    for (int i = 0; i < text_len; i++) vector[i] = (double)text[i];
}

// Chargement des données d'entraînement
int load_train_data(double *input_data, double *target_data, int len) {
    FILE *file = fopen(DATABASE_FILE, "r");
    if (!file) { perror("Erreur ouverture fichier"); return 0; }
    char line[512]; int count = 0;
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

// Test du modèle
void test_model(const char *question) {
    double input_test[DIMENSION] = {0};
    double output_test[DIMENSION] = {0};
    encode_text_to_vector(question, input_test, DIMENSION);
    multi_head_attention(input_test, input_test, input_test, output_test, DIMENSION);
    printf("Réponse prédite : ");
    for (int i = 0; i < DIMENSION; i++) {
        char c = (char)round(output_test[i]);
        if (c < 32 || c > 126) c = '?';
        printf("%c", c);
    }
    printf("\n");
}

// Fonction pour rechercher une sous-chaîne sans tenir compte de la casse
char *strcasestr_custom(const char *haystack, const char *needle) {
    if (!haystack || !needle) return NULL;

    size_t len_h = strlen(haystack);
    size_t len_n = strlen(needle);

    if (len_n > len_h) return NULL; // Sécurité pour éviter dépassement

    for (size_t i = 0; i <= len_h - len_n; i++) {
        if (strncasecmp(haystack + i, needle, len_n) == 0) {
            return (char *)(haystack + i);
        }
    }

    return NULL;
}


// Vérifier si l'utilisateur se présente et extraire son prénom
int detect_name(char *input) {
    char *phrases[] = {"moi c'est ", "je suis ", "mon nom est ","je me nomme ","je me presente "};
    int num_phrases = 5;

    for (int i = 0; i < num_phrases; i++) {
        char *pos = strcasestr_custom(input, phrases[i]);  // Recherche insensible à la casse
        if (pos != NULL) {
            char *name = pos + strlen(phrases[i]);
            printf("Réponse : Enchanté %s !\n", name);
            return 1;
        }
    }

    return 0;
}


void chat() {
    char input[256];

    while (1) {
        printf("\nVous : ");
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "exit") == 0) {
            printf("Fin de la conversation.\n");
            break;
        }

        if (!detect_name(input)) {
            test_model(input);
        }
    }
}

int main() {
    double input_data[MAX_ENTRIES * DIMENSION] = {0};
    double target_data[MAX_ENTRIES * DIMENSION] = {0};
    int num_samples = load_train_data(input_data, target_data, DIMENSION);
    printf("%d entrainements chargés\n", num_samples);

    // test_model("Bonjour");
    chat();
    return 0;
}
