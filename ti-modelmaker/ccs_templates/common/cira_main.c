/**
 * CiRA ME - TI MCU Inference Template
 *
 * This file provides the main entry point for running a CiRA ME exported
 * model on TI C2000 MCUs. It handles:
 *   - Device initialization
 *   - SCI (UART) serial communication for receiving features and sending predictions
 *   - Model inference using the exported emlearn C header
 *
 * Protocol (binary, little-endian):
 *   PC -> MCU:  [0xAA] [num_features:uint8] [feature_1:float32] ... [feature_N:float32]
 *   MCU -> PC:  [0xBB] [prediction:float32] [inference_time_us:uint32]
 *
 * Usage:
 *   1. Open this project in Code Composer Studio
 *   2. Replace "cira_model.h" with your exported model header
 *   3. Update MODEL_NUM_FEATURES to match your model
 *   4. Build and flash to your C2000 LaunchPad
 *   5. Connect via serial terminal at 115200 baud
 *
 * Compatible devices: F28379D, F280049C, F28P55x
 */

#include "driverlib.h"
#include "device.h"

/* ============================================================
 * USER CONFIGURATION - Update these for your model
 * ============================================================ */

// Include your CiRA ME exported model header
#include "cira_model.h"

// Number of input features your model expects
#define MODEL_NUM_FEATURES      6

// Maximum features supported (memory budget)
#define MAX_FEATURES            64

// Serial baud rate
#define SCI_BAUDRATE            115200

// Protocol markers
#define PROTO_START_INPUT       0xAA
#define PROTO_START_OUTPUT      0xBB
#define PROTO_START_TEXT        0xCC
#define PROTO_CMD_PING          0x01
#define PROTO_CMD_INFO          0x02

/* ============================================================
 * GLOBAL VARIABLES
 * ============================================================ */
static float g_features[MAX_FEATURES];
static float g_prediction;
static volatile uint32_t g_timer_count = 0;

/* ============================================================
 * SCI (UART) FUNCTIONS
 * ============================================================ */

/**
 * Initialize SCI-A for UART communication.
 * GPIO pins depend on the target device - see device-specific section.
 */
void CIRA_SCI_init(void)
{
    // GPIO configuration is device-specific
    // Handled by CIRA_Device_initPins() below

    SCI_performSoftwareReset(SCIA_BASE);
    SCI_setConfig(SCIA_BASE, DEVICE_LSPCLK_FREQ, SCI_BAUDRATE,
                  (SCI_CONFIG_WLEN_8 | SCI_CONFIG_STOP_ONE |
                   SCI_CONFIG_PAR_NONE));
    SCI_resetChannels(SCIA_BASE);
    SCI_enableFIFO(SCIA_BASE);
    SCI_enableModule(SCIA_BASE);
    SCI_performSoftwareReset(SCIA_BASE);
}

/**
 * Send a single byte via SCI.
 */
void CIRA_SCI_sendByte(uint16_t data)
{
    while(SCI_getTxFIFOStatus(SCIA_BASE) == SCI_FIFO_TX16) {}
    SCI_writeCharNonBlocking(SCIA_BASE, data & 0xFF);
}

/**
 * Receive a single byte via SCI (blocking).
 */
uint16_t CIRA_SCI_recvByte(void)
{
    while(SCI_getRxFIFOStatus(SCIA_BASE) == SCI_FIFO_RX0) {}
    return SCI_readCharBlockingFIFO(SCIA_BASE) & 0xFF;
}

/**
 * Send a float as 4 bytes (little-endian).
 */
void CIRA_SCI_sendFloat(float value)
{
    uint16_t *bytes = (uint16_t *)&value;
    // C2000 is 16-bit word addressable, float = 2 words
    CIRA_SCI_sendByte(bytes[0] & 0xFF);
    CIRA_SCI_sendByte((bytes[0] >> 8) & 0xFF);
    CIRA_SCI_sendByte(bytes[1] & 0xFF);
    CIRA_SCI_sendByte((bytes[1] >> 8) & 0xFF);
}

/**
 * Receive a float from 4 bytes (little-endian).
 */
float CIRA_SCI_recvFloat(void)
{
    union {
        float f;
        uint16_t words[2];
    } val;

    uint16_t b0 = CIRA_SCI_recvByte();
    uint16_t b1 = CIRA_SCI_recvByte();
    uint16_t b2 = CIRA_SCI_recvByte();
    uint16_t b3 = CIRA_SCI_recvByte();

    val.words[0] = b0 | (b1 << 8);
    val.words[1] = b2 | (b3 << 8);
    return val.f;
}

/**
 * Send a uint32 as 4 bytes (little-endian).
 */
void CIRA_SCI_sendUint32(uint32_t value)
{
    CIRA_SCI_sendByte(value & 0xFF);
    CIRA_SCI_sendByte((value >> 8) & 0xFF);
    CIRA_SCI_sendByte((value >> 16) & 0xFF);
    CIRA_SCI_sendByte((value >> 24) & 0xFF);
}

/**
 * Send a null-terminated string via SCI.
 */
void CIRA_SCI_sendString(const char *str)
{
    while(*str) {
        CIRA_SCI_sendByte(*str++);
    }
}

/* ============================================================
 * TIMER FUNCTIONS (for measuring inference time)
 * ============================================================ */

void CIRA_Timer_init(void)
{
    CPUTimer_stopTimer(CPUTIMER0_BASE);
    CPUTimer_setPreScaler(CPUTIMER0_BASE, 0);
    CPUTimer_setPeriod(CPUTIMER0_BASE, 0xFFFFFFFF);
    CPUTimer_startTimer(CPUTIMER0_BASE);
}

uint32_t CIRA_Timer_getCount(void)
{
    return CPUTimer_getTimerCount(CPUTIMER0_BASE);
}

/* ============================================================
 * MODEL INFERENCE
 * ============================================================ */

/**
 * Run model inference on the feature buffer.
 * Returns prediction as float.
 * For classification: returns class index as float.
 * For regression: returns predicted value.
 */
float CIRA_predict(float *features, uint16_t num_features)
{
    // emlearn models use this signature:
    // For regression: float model_predict(const float *features, int n_features)
    // The function name matches the model header filename
    return cira_model_predict(features, num_features);
}

/* ============================================================
 * PROTOCOL HANDLER
 * ============================================================ */

void CIRA_handleCommand(void)
{
    uint16_t startByte = CIRA_SCI_recvByte();

    if(startByte == PROTO_START_INPUT)
    {
        // Receive feature count
        uint16_t num_features = CIRA_SCI_recvByte();
        if(num_features > MAX_FEATURES) num_features = MAX_FEATURES;

        // Receive features
        uint16_t i;
        for(i = 0; i < num_features; i++)
        {
            g_features[i] = CIRA_SCI_recvFloat();
        }

        // Measure inference time
        uint32_t t_start = CIRA_Timer_getCount();

        // Run inference
        g_prediction = CIRA_predict(g_features, num_features);

        uint32_t t_end = CIRA_Timer_getCount();
        // Timer counts DOWN, so elapsed = start - end
        uint32_t elapsed_cycles = t_start - t_end;
        // Convert to microseconds (assuming DEVICE_SYSCLK_FREQ)
        uint32_t elapsed_us = elapsed_cycles / (DEVICE_SYSCLK_FREQ / 1000000UL);

        // Send response
        CIRA_SCI_sendByte(PROTO_START_OUTPUT);
        CIRA_SCI_sendFloat(g_prediction);
        CIRA_SCI_sendUint32(elapsed_us);
    }
    else if(startByte == PROTO_CMD_PING)
    {
        // Ping response
        CIRA_SCI_sendByte(PROTO_CMD_PING);
        CIRA_SCI_sendString("CiRA-ME-OK\r\n");
    }
    else if(startByte == PROTO_CMD_INFO)
    {
        // Send model info
        CIRA_SCI_sendByte(PROTO_START_TEXT);
        CIRA_SCI_sendString("CiRA ME Inference Engine\r\n");
        CIRA_SCI_sendString("Features: ");
        // Send feature count as ASCII
        CIRA_SCI_sendByte('0' + (MODEL_NUM_FEATURES / 10));
        CIRA_SCI_sendByte('0' + (MODEL_NUM_FEATURES % 10));
        CIRA_SCI_sendString("\r\n");
    }
}

/* ============================================================
 * DEVICE-SPECIFIC PIN INITIALIZATION
 * Override this in device-specific main if needed
 * ============================================================ */

#ifndef CIRA_CUSTOM_PINS
void CIRA_Device_initPins(void)
{
    // Default: F28379D LaunchPad SCI-A pins
    // GPIO28 = SCIRXDA, GPIO29 = SCITXDA
    GPIO_setPinConfig(GPIO_28_SCIA_RX);
    GPIO_setPinConfig(GPIO_29_SCIA_TX);
    GPIO_setDirectionMode(28, GPIO_DIR_MODE_IN);
    GPIO_setDirectionMode(29, GPIO_DIR_MODE_OUT);
    GPIO_setPadConfig(28, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(29, GPIO_PIN_TYPE_STD);
    GPIO_setQualificationMode(28, GPIO_QUAL_ASYNC);
}
#endif

/* ============================================================
 * MAIN
 * ============================================================ */

void main(void)
{
    // Initialize device clocks and peripherals
    Device_init();
    Device_initGPIO();

    // Initialize SCI pins (device-specific)
    CIRA_Device_initPins();

    // Initialize SCI (UART) at 115200 baud
    CIRA_SCI_init();

    // Initialize timer for inference timing
    CIRA_Timer_init();

    // Enable global interrupts
    EINT;
    ERTM;

    // Send startup message
    CIRA_SCI_sendString("\r\n=== CiRA ME Inference Engine ===\r\n");
    CIRA_SCI_sendString("Ready. Waiting for input...\r\n");

    // Main inference loop
    for(;;)
    {
        CIRA_handleCommand();
    }
}
