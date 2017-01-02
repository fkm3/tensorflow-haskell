#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

// This file has been modified from the original to allow it to work with c2hs.

#include <stdbool.h>

typedef struct ALEInterface {
} ALEInterface;
typedef struct ALEState {
} ALEState;

// Declares int rgb_palette[256]
ALEInterface *ALE_new();
void ALE_del(ALEInterface *ale);
const char *getString(ALEInterface *ale, const char *key);
int getInt(ALEInterface *ale, const char *key);
bool getBool(ALEInterface *ale, const char *key);
float getFloat(ALEInterface *ale, const char *key);
void setString(ALEInterface *ale, const char *key, const char *value);
void setInt(ALEInterface *ale, const char *key, int value);
void setBool(ALEInterface *ale, const char *key, bool value);
void setFloat(ALEInterface *ale, const char *key, float value);
void loadROM(ALEInterface *ale, const char *rom_file);
int act(ALEInterface *ale, int action);
bool game_over(ALEInterface *ale);
void reset_game(ALEInterface *ale);
void getLegalActionSet(ALEInterface *ale, int *actions);
int getLegalActionSize(ALEInterface *ale);
void getMinimalActionSet(ALEInterface *ale, int *actions);
int getMinimalActionSize(ALEInterface *ale);
int getFrameNumber(ALEInterface *ale);
int lives(ALEInterface *ale);
int getEpisodeFrameNumber(ALEInterface *ale);
void getScreen(ALEInterface *ale, unsigned char *screen_data);
void getRAM(ALEInterface *ale, unsigned char *ram);
int getRAMSize(ALEInterface *ale);
int getScreenWidth(ALEInterface *ale);
int getScreenHeight(ALEInterface *ale);
void getScreenRGB(ALEInterface *ale, unsigned char *output_buffer);
void getScreenGrayscale(ALEInterface *ale, unsigned char *output_buffer);
void saveState(ALEInterface *ale);
void loadState(ALEInterface *ale);
ALEState *cloneState(ALEInterface *ale);
void restoreState(ALEInterface *ale, ALEState *state);
ALEState *cloneSystemState(ALEInterface *ale);
void restoreSystemState(ALEInterface *ale, ALEState *state);
void deleteState(ALEState *state);
void saveScreenPNG(ALEInterface *ale, const char *filename);

// Encodes the state as a raw bytestream. This may have multiple '\0'
// characters and thus should not be treated as a C string. Use encodeStateLen
// to find the length of the buffer to pass in, or it will be overrun as this
// simply memcpys bytes into the buffer.
void encodeState(ALEState *state, char *buf, int buf_len);
int encodeStateLen(ALEState *state);
ALEState *decodeState(const char *serialized, int len);

// 0: Info, 1: Warning, 2: Error
void setLoggerMode(int mode);

#endif
