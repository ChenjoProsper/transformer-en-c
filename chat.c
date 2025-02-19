#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_ENTRIES 10000  // Nombre maximal d'entrées dans la base de données
#define MAX_LENGTH 256   // Taille maximale d'une ligne de la base de données

typedef struct {
    char question[MAX_LENGTH];
    char response[MAX_LENGTH];
} QAEntry;

QAEntry database[MAX_ENTRIES];
int db_size = 0;

// Fonction pour calculer la distance de Levenshtein
int levenshtein_distance(const char *s1, const char *s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);

    int **matrix = (int **)malloc((len1 + 1) * sizeof(int *));
    for (int i = 0; i <= len1; i++) {
        matrix[i] = (int *)malloc((len2 + 1) * sizeof(int));
    }

    for (int i = 0; i <= len1; i++) matrix[i][0] = i;
    for (int j = 0; j <= len2; j++) matrix[0][j] = j;

    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            matrix[i][j] = fmin(fmin(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1), matrix[i - 1][j - 1] + cost);
        }
    }

    int result = matrix[len1][len2];

    // Libérer la mémoire
    for (int i = 0; i <= len1; i++) {
        free(matrix[i]);
    }
    free(matrix);

    return result;
}


// Charger la base de données à partir du fichier
void load_text_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        exit(1);
    }

    char line[MAX_LENGTH];
    db_size = 0; // Réinitialiser la taille de la base de données

    while (fgets(line, sizeof(line), file) != NULL) {
        line[strcspn(line, "\n")] = 0; // Supprimer le saut de ligne

        char *delimiter = strchr(line, '|'); // Trouver le séparateur
        if (delimiter != NULL) {
            *delimiter = '\0'; // Séparer question et réponse
            strncpy(database[db_size].question, line, MAX_LENGTH);
            strncpy(database[db_size].response, delimiter + 1, MAX_LENGTH);
            db_size++;
        }

        if (db_size >= MAX_ENTRIES) break; // Limite atteinte
    }

    fclose(file);

    if (db_size == 0) {
        printf("⚠️ Base de données vide ! Ajoutez des entrées dans %s.\n", filename);
    } else {
        printf("✅ Base de données chargée (%d entrées)\n", db_size);
    }
}


// Ajouter une nouvelle entrée à la base de données et au fichier
void add_to_database(const char *question, const char *response, const char *filename) {
    if (db_size >= MAX_ENTRIES) {
        printf("La base de données est pleine, impossible d'ajouter de nouvelles entrées.\n");
        return;
    }

    // Ajouter à la mémoire du chatbot
    strncpy(database[db_size].question, question, MAX_LENGTH);
    strncpy(database[db_size].response, response, MAX_LENGTH);
    db_size++;

    // Ajouter au fichier pour sauvegarde
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        return;
    }

    fprintf(file, "%s|%s\n", question, response);
    fclose(file);

    printf("✅ Réponse apprise et ajoutée à la base de données !\n");
}

// Modifier la fonction pour apprendre
void generate_response(char *input, const char *filename) {
    if (db_size == 0) {
        printf("Erreur : Base de données vide !\n");
        return;
    }

    int min_distance = MAX_LENGTH;
    int best_match_index = -1;

    for (int i = 0; i < db_size; i++) {
        int dist = levenshtein_distance(input, database[i].question);
        if (dist < min_distance) {
            min_distance = dist;
            best_match_index = i;
        }
    }

    if (best_match_index != -1 && min_distance <= 5) { // Tolérance d'erreur
        printf("Réponse : %s\n", database[best_match_index].response);
    } else {
        // Le chatbot ne sait pas répondre, il apprend
        printf("Je ne sais pas quoi répondre... Que dois-je dire ?\nVous : ");
        char new_response[MAX_LENGTH];
        fgets(new_response, sizeof(new_response), stdin);
        new_response[strcspn(new_response, "\n")] = 0;

        add_to_database(input, new_response, filename);
    }
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

void chat(const char *filename) {
    char input[MAX_LENGTH];

    while (1) {
        printf("\nVous : ");
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "exit") == 0) {
            printf("Fin de la conversation.\n");
            break;
        }

        if (!detect_name(input)) {
            generate_response(input,filename);
        }
    }
}

int main() {
    const char *filename = "database.txt";

    // Charger la base de données
    load_text_file(filename);

    // Lancer la conversation
    chat(filename);

    return 0;
}

