#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define CHUNK_SIZE 1000
#define MAX_LINE_LENGTH 2048

typedef struct {
    char **lines;
    int count;
} Chunk;

// Event types
typedef enum {
    PREPROCESSING_STARTED,
    CHUNK_PROCESSED,
    PREPROCESSING_COMPLETED
} EventType;

// Event handler function prototype
typedef void (*EventHandler)(EventType event, double metric, double throughput);

// Function to remove non-ASCII characters from a string
void clean_text(char *text) {
    int j = 0;
    for (int i = 0; text[i] != '\0'; i++) {
        if ((unsigned char)text[i] < 128) {  // ASCII characters only (0-127)
            text[j++] = text[i];
        }
    }
    text[j] = '\0';
}

// Function to process a chunk of data
void process_chunk(Chunk *chunk, FILE *output_file, EventHandler event_handler) {
    double start_time = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < chunk->count; i++) {
        clean_text(chunk->lines[i]);  // Clean each line in parallel
    }

    // Write cleaned data to output file
    #pragma omp critical
    {
        for (int i = 0; i < chunk->count; i++) {
            fputs(chunk->lines[i], output_file);
        }
    }

    double chunk_time = omp_get_wtime() - start_time;
    double chunk_throughput = chunk->count / chunk_time; // Calculate throughput for this chunk
    event_handler(CHUNK_PROCESSED, chunk_time, chunk_throughput);  // Trigger event for chunk processed
}

// Function to read a chunk of data from the input file
int read_chunk(FILE *file, Chunk *chunk) {
    chunk->lines = malloc(CHUNK_SIZE * sizeof(char *));
    chunk->count = 0;

    while (chunk->count < CHUNK_SIZE) {
        char *line = malloc(MAX_LINE_LENGTH * sizeof(char));
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            free(line);
            break;
        }
        chunk->lines[chunk->count++] = line;
    }

    return chunk->count;
}

// Event handler function
void handle_event(EventType event, double metric, double throughput) {
    static double total_throughput = 0;
    static int chunk_counter = 0;

    switch (event) {
        case PREPROCESSING_STARTED:
            printf("Preprocessing started...\n");
            break;
        case CHUNK_PROCESSED:
            printf("Processed chunk %d in %.2f seconds with throughput: %.2f records/sec\n", ++chunk_counter, metric, throughput);
            total_throughput += throughput;
            break;
        case PREPROCESSING_COMPLETED:
            printf("Preprocessing completed. Total processing time: %.2f seconds.\n", metric);
            printf("Average throughput: %.2f records/sec\n", total_throughput / chunk_counter);
            break;
        default:
            printf("Unknown event.\n");
    }
}

// Main function
int main() {
    FILE *input_file = fopen("large_mobiles.csv", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error opening input file.\n");
        return 1;
    }

    FILE *output_file = fopen("cleaned_mobiles_data.csv", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file.\n");
        fclose(input_file);
        return 1;
    }

    // Start preprocessing
    double total_start_time = omp_get_wtime();
    handle_event(PREPROCESSING_STARTED, 0, 0);

    int chunk_counter = 0;
    while (1) {
        Chunk chunk;
        int records_read = read_chunk(input_file, &chunk);

        if (records_read == 0) break;  // No more records to read

        // Process each chunk with event handling
        process_chunk(&chunk, output_file, handle_event);

        // Free memory allocated for the chunk
        for (int i = 0; i < chunk.count; i++) {
            free(chunk.lines[i]);
        }
        free(chunk.lines);

        chunk_counter++;
    }

    // Complete preprocessing
    double total_time = omp_get_wtime() - total_start_time;
    handle_event(PREPROCESSING_COMPLETED, total_time, 0);

    fclose(input_file);
    fclose(output_file);

    printf("Data cleaning completed. Cleaned data saved to cleaned_mobiles_data.csv\n");

    return 0;
}
