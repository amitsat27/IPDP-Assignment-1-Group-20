#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#define CHUNK_SIZE 1000      // Number of lines to process per chunk
#define MAX_LINE_LENGTH 6000  // Adjusted to handle larger lines with 50 columns

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

// Thread data structure
typedef struct {
    int start_index;
    int end_index;
    Chunk *chunk;
    FILE *output_file;
    EventHandler event_handler;
} ThreadData;

// Mutex for critical section (writing to the output file)
pthread_mutex_t write_mutex = PTHREAD_MUTEX_INITIALIZER;

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

// Function to process a line of data in parallel using pthreads
void *process_lines(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start_index; i < data->end_index; i++) {
        clean_text(data->chunk->lines[i]);  // Clean the specific line
    }
    return NULL;
}



void preprocessing_completed(double total_time, double avg_throughput) {
    printf("\n");
    printf("**********************************************************\n");
    printf("*      Preprocessing Completed                           *\n");
    printf("*      Total processing time: %.2f sec                   *\n", total_time);
    printf("*      Average throughput: %.2f records/sec           *\n", avg_throughput);
    printf("**********************************************************\n");
    printf("\n");
}

// Function to process the entire chunk of data
void process_chunk(Chunk *chunk, FILE *output_file, EventHandler event_handler, int num_threads) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);  // Start timer

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));  // Allocate memory for threads
    ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));  // Allocate memory for thread data

    // Distribute work across the threads
    int lines_per_thread = (chunk->count + num_threads - 1) / num_threads;  // Divide lines among threads, rounding up

    for (int i = 0; i < num_threads; i++) {
        int start_line = i * lines_per_thread;
        int end_line = (i + 1) * lines_per_thread;
        if (end_line > chunk->count) end_line = chunk->count;

        // Prepare data for each thread
        thread_data[i].start_index = start_line;
        thread_data[i].end_index = end_line;
        thread_data[i].chunk = chunk;
        thread_data[i].output_file = output_file;
        thread_data[i].event_handler = event_handler;

        // Create thread to process lines in the chunk
        pthread_create(&threads[i], NULL, process_lines, &thread_data[i]);
    }

    // Wait for all threads to finish processing
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Write cleaned data to the output file in a critical section
    pthread_mutex_lock(&write_mutex);
    for (int i = 0; i < chunk->count; i++) {
        fputs(chunk->lines[i], output_file);
    }
    pthread_mutex_unlock(&write_mutex);

    // Measure elapsed time
    gettimeofday(&end_time, NULL);
    double chunk_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;
    double chunk_throughput = chunk->count / chunk_time; // Throughput (records per second)
    event_handler(CHUNK_PROCESSED, chunk_time, chunk_throughput);

    free(threads);  // Free allocated memory
    free(thread_data);  // Free allocated memory
}

// Function to read a chunk of data from the input file
int read_chunk(FILE *file, Chunk *chunk) {
    chunk->lines = malloc(CHUNK_SIZE * sizeof(char *));  // Allocate memory for chunk lines
    chunk->count = 0;

    while (chunk->count < CHUNK_SIZE) {
        char *line = malloc(MAX_LINE_LENGTH * sizeof(char));  // Allocate memory for each line
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            free(line);  // Free line memory if EOF reached
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
            printf("*          Processing Started            *\n");
            break;
        case CHUNK_PROCESSED:
            printf("Processed chunk %d in %.2f seconds with throughput: %.2f records/sec\n", ++chunk_counter, metric, throughput);
            total_throughput += throughput;
            break;
        case PREPROCESSING_COMPLETED:
            preprocessing_completed(metric, total_throughput / chunk_counter);
            break;
        default:
            printf("Unknown event.\n");
    }
}


// Main function
int main() {
    // Set the number of threads directly in the program
    int num_threads = 256;

    FILE *input_file = fopen("NoisyMobileDataLight.csv", "r");
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
    struct timeval total_start_time, total_end_time;
    gettimeofday(&total_start_time, NULL);  // Start total timer
    handle_event(PREPROCESSING_STARTED, 0, 0);

    int chunk_counter = 0;
    while (1) {
        Chunk chunk;
        int records_read = read_chunk(input_file, &chunk);

        if (records_read == 0) break;  // No more records to read

        // Process each chunk with event handling
        process_chunk(&chunk, output_file, handle_event, num_threads);

        // Free memory allocated for the chunk
        for (int i = 0; i < chunk.count; i++) {
            free(chunk.lines[i]);  // Free each line
        }
        free(chunk.lines);  // Free chunk lines array

        chunk_counter++;
    }

    // Complete preprocessing
    gettimeofday(&total_end_time, NULL);  // End total timer
    double total_time = (total_end_time.tv_sec - total_start_time.tv_sec) + (total_end_time.tv_usec - total_start_time.tv_usec) / 1e6;
    handle_event(PREPROCESSING_COMPLETED, total_time, 0);

    fclose(input_file);
    fclose(output_file);

    printf("Data cleaning completed. Cleaned data saved to cleaned_mobiles_data.csv\n");

    return 0;
}
