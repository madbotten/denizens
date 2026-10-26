#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int x;
    int y;
} Point2D;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint8_t *loadRGBA(const char *path, size_t *outW, size_t *outH) {
    NSString *inPath = [NSString stringWithUTF8String:path];
    NSURL *inURL = [NSURL fileURLWithPath:inPath];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)inURL, NULL);
    if (!src) return NULL;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!img) return NULL;

    size_t w = CGImageGetWidth(img);
    size_t h = CGImageGetHeight(img);
    size_t stride = w * 4;
    uint8_t *pixels = (uint8_t *)calloc(h * stride, 1);

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        pixels, w, h, 8, stride, cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(img);

    *outW = w;
    *outH = h;
    return pixels;
}

static bool saveRGBA(const char *path, uint8_t *pixels, size_t w, size_t h) {
    size_t stride = w * 4;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, pixels, h * stride, NULL);
    CGImageRef outImg = CGImageCreate(
        w, h, 8, 32, stride, cs,
        kCGImageAlphaLast | kCGBitmapByteOrder32Big,
        provider, NULL, false, kCGRenderingIntentDefault
    );

    NSString *outPath = [NSString stringWithUTF8String:path];
    NSURL *outURL = [NSURL fileURLWithPath:outPath];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)outURL, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL
    );
    if (!dest) {
        CGImageRelease(outImg);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(cs);
        return false;
    }
    CGImageDestinationAddImage(dest, outImg, NULL);
    bool ok = CGImageDestinationFinalize(dest);

    CFRelease(dest);
    CGImageRelease(outImg);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(cs);
    return ok;
}

static int runCompositeFrame(const char *masterFramePath, const char *targetPath, const char *outPath) {
    size_t mw = 0, mh = 0, tw = 0, th = 0;
    uint8_t *master = loadRGBA(masterFramePath, &mw, &mh);
    uint8_t *target = loadRGBA(targetPath, &tw, &th);
    if (!master || !target || mw != tw || mh != th) {
        fprintf(stderr, "Failed to load matching dimensions for frame composite\n");
        free(master);
        free(target);
        return 1;
    }

    float borderOuter = 44.0f * ((float)tw / 1024.0f);
    float borderInner = 64.0f * ((float)tw / 1024.0f);
    float cornerOuter = 152.0f * ((float)tw / 1024.0f);
    float cornerInner = 185.0f * ((float)tw / 1024.0f);

    for (size_t y = 0; y < th; y++) {
        for (size_t x = 0; x < tw; x++) {
            float dx = fminf((float)x, (float)(tw - 1 - x));
            float dy = fminf((float)y, (float)(th - 1 - y));
            float edgeDist = fminf(dx, dy);
            float cornerDist = dx + dy;

            float wEdge = 1.0f - clampf((edgeDist - borderOuter) / (borderInner - borderOuter), 0.0f, 1.0f);
            float wCorner = 1.0f - clampf((cornerDist - cornerOuter) / (cornerInner - cornerOuter), 0.0f, 1.0f);
            float weight = fmaxf(wEdge, wCorner);

            size_t idx = (y * tw + x) * 4;
            if (weight > 0.001f) {
                for (int c = 0; c < 3; c++) {
                    float mVal = master[idx + c];
                    float tVal = target[idx + c];
                    target[idx + c] = (uint8_t)lrintf(mVal * weight + tVal * (1.0f - weight));
                }
            }
            target[idx + 3] = 255;
        }
    }

    bool ok = saveRGBA(outPath, target, tw, th);
    free(master);
    free(target);
    return ok ? 0 : 1;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc >= 5 && strcmp(argv[1], "--composite-frame") == 0) {
            return runCompositeFrame(argv[2], argv[3], argv[4]);
        }
        if (argc < 3) {
            fprintf(stderr, "Usage:\n  %s <input_image> <output_png>\n  %s --composite-frame <master_frame> <input_portrait> <output_png>\n", argv[0], argv[0]);
            return 1;
        }

        size_t w = 0, h = 0;
        uint8_t *pixels = loadRGBA(argv[1], &w, &h);
        if (!pixels) {
            fprintf(stderr, "Failed to decode %s\n", argv[1]);
            return 1;
        }

        // 1. Estimate mean border background color (cool slate gray ~ RGB(38, 47, 55))
        double sumR = 0, sumG = 0, sumB = 0;
        size_t countBorder = 0;
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                if (x < 8 || x >= w - 8 || y < 8 || y >= h - 8) {
                    size_t idx = (y * w + x) * 4;
                    sumR += pixels[idx + 0];
                    sumG += pixels[idx + 1];
                    sumB += pixels[idx + 2];
                    countBorder++;
                }
            }
        }
        float meanBgR = (float)(sumR / countBorder);
        float meanBgG = (float)(sumG / countBorder);
        float meanBgB = (float)(sumB / countBorder);

        // Compute normalized background chromaticity vector
        float bgLen = sqrtf(meanBgR*meanBgR + meanBgG*meanBgG + meanBgB*meanBgB) + 1e-5f;
        float bgUnitR = meanBgR / bgLen;
        float bgUnitG = meanBgG / bgLen;
        float bgUnitB = meanBgB / bgLen;

        // Compute local Sobel gradient magnitude
        float *grad = (float *)calloc(w * h, sizeof(float));
        for (size_t y = 1; y + 1 < h; y++) {
            for (size_t x = 1; x + 1 < w; x++) {
                float gx = 0, gy = 0;
                for (int c = 0; c < 3; c++) {
                    float tl = pixels[((y - 1) * w + (x - 1)) * 4 + c];
                    float tc = pixels[((y - 1) * w + x) * 4 + c];
                    float tr = pixels[((y - 1) * w + (x + 1)) * 4 + c];
                    float ml = pixels[(y * w + (x - 1)) * 4 + c];
                    float mr = pixels[(y * w + (x + 1)) * 4 + c];
                    float bl = pixels[((y + 1) * w + (x - 1)) * 4 + c];
                    float bc = pixels[((y + 1) * w + x) * 4 + c];
                    float br = pixels[((y + 1) * w + (x + 1)) * 4 + c];
                    float sx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
                    float sy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
                    gx += sx * sx;
                    gy += sy * sy;
                }
                grad[y * w + x] = sqrtf((gx + gy) / 3.0f) * 0.25f;
            }
        }

        // Helper criterion: is a pixel part of the cool blue-slate background (or soft slate vignette/shadow)?
        // Notice on all 9 images: slate background has B > G > R, specifically (B - G) >= 3 and (B - R) >= 8,
        // whereas every green/olive/bronze/gold scale on the snake has G >= B - 1 or R >= B!
        float *bgR = (float *)calloc(w * h, sizeof(float));
        float *bgG = (float *)calloc(w * h, sizeof(float));
        float *bgB = (float *)calloc(w * h, sizeof(float));
        uint8_t *visited = (uint8_t *)calloc(w * h, 1);
        uint8_t *isBgRegion = (uint8_t *)calloc(w * h, 1);
        Point2D *queue = (Point2D *)malloc(w * h * sizeof(Point2D));
        size_t qHead = 0, qTail = 0;

        // Seed outer border first and flood-fill exterior background
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
                    size_t i = y * w + x;
                    visited[i] = 1;
                    isBgRegion[i] = 1;
                    bgR[i] = pixels[i * 4 + 0];
                    bgG[i] = pixels[i * 4 + 1];
                    bgB[i] = pixels[i * 4 + 2];
                    queue[qTail++] = (Point2D){(int)x, (int)y};
                }
            }
        }

        const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

        while (qHead < qTail) {
            Point2D p = queue[qHead++];
            size_t curIdx = (size_t)p.y * w + (size_t)p.x;
            float refR = bgR[curIdx];
            float refG = bgG[curIdx];
            float refB = bgB[curIdx];

            for (int dir = 0; dir < 8; dir++) {
                int nx = p.x + dx[dir];
                int ny = p.y + dy[dir];
                if (nx < 0 || nx >= (int)w || ny < 0 || ny >= (int)h) continue;
                size_t nIdx = (size_t)ny * w + (size_t)nx;
                if (visited[nIdx]) continue;

                float r = pixels[nIdx * 4 + 0];
                float g = pixels[nIdx * 4 + 1];
                float b = pixels[nIdx * 4 + 2];

                float stepDist = sqrtf((r - refR)*(r - refR) + (g - refG)*(g - refG) + (b - refB)*(b - refB));
                float globalDist = sqrtf((r - meanBgR)*(r - meanBgR) + (g - meanBgG)*(g - meanBgG) + (b - meanBgB)*(b - meanBgB));

                bool hasSlateChroma = ((b - g) >= 1.5f && (b - r) >= 2.5f && b >= 15.0f && b <= meanBgB + 16.0f);
                bool isSmoothSlate = hasSlateChroma && (stepDist < 8.5f) && (globalDist < 24.0f) && (grad[nIdx] < 11.0f);
                // Also traverse soft dark ambient glow / shadow in the background outside the crisp black ink outline (which has r+g+b < 32)
                bool isSoftAmbientBg = (stepDist < 12.0f) && (grad[nIdx] < 16.0f) && ((r + g + b) >= 54.0f) && (r < 110.0f && g < 118.0f && b < 118.0f) && ((r + g + b) < 240.0f);
                // Also traverse flat dark slate drop-shadow bands (~RGB(14,17,18), sum 42..68) outside the black ink outline (sum < 32)
                bool isDropShadowBg = ((r + g + b) >= 42.0f && (r + g + b) <= 68.0f) &&
                                      ((b - r) >= 0.0f && (b - r) <= 7.0f) &&
                                      ((b - g) >= -1.5f && (b - g) <= 5.0f) &&
                                      (stepDist < 28.0f) && (grad[nIdx] < 24.0f);

                if (isSmoothSlate || isSoftAmbientBg || isDropShadowBg) {
                    visited[nIdx] = 1;
                    isBgRegion[nIdx] = 1;
                    bgR[nIdx] = refR * 0.85f + r * 0.15f;
                    bgG[nIdx] = refG * 0.85f + g * 0.15f;
                    bgB[nIdx] = refB * 0.85f + b * 0.15f;
                    queue[qTail++] = (Point2D){nx, ny};
                }
            }
        }

        // Seed interior enclosed holes ONLY where a full 5x5 patch is ultra-flat background matching meanBg
        float meanBGDiff = meanBgB - meanBgG;
        float meanBRDiff = meanBgB - meanBgR;
        for (size_t y = 4; y + 4 < h; y++) {
            for (size_t x = 4; x + 4 < w; x++) {
                size_t centerIdx = y * w + x;
                if (isBgRegion[centerIdx]) continue;

                bool patchIsFlatBg = true;
                for (int py = -2; py <= 2 && patchIsFlatBg; py++) {
                    for (int px = -2; px <= 2; px++) {
                        size_t idx = (y + py) * w + (x + px);
                        float r = pixels[idx * 4 + 0];
                        float g = pixels[idx * 4 + 1];
                        float b = pixels[idx * 4 + 2];
                        float dMean = sqrtf((r - meanBgR)*(r - meanBgR) + (g - meanBgG)*(g - meanBgG) + (b - meanBgB)*(b - meanBgB));
                        if (dMean > 4.5f ||
                            fabsf((b - g) - meanBGDiff) > 2.5f ||
                            fabsf((b - r) - meanBRDiff) > 3.5f ||
                            (b - g) < 1.5f || (b - r) < 4.0f ||
                            grad[idx] > 3.8f) {
                            patchIsFlatBg = false;
                            break;
                        }
                    }
                }
                if (patchIsFlatBg && !visited[centerIdx]) {
                    visited[centerIdx] = 1;
                    isBgRegion[centerIdx] = 1;
                    bgR[centerIdx] = pixels[centerIdx * 4 + 0];
                    bgG[centerIdx] = pixels[centerIdx * 4 + 1];
                    bgB[centerIdx] = pixels[centerIdx * 4 + 2];
                    queue[qTail++] = (Point2D){(int)x, (int)y};
                }
            }
        }

        // Flood-fill from the verified 5x5 flat interior background seeds
        while (qHead < qTail) {
            Point2D p = queue[qHead++];
            size_t curIdx = (size_t)p.y * w + (size_t)p.x;
            float refR = bgR[curIdx];
            float refG = bgG[curIdx];
            float refB = bgB[curIdx];

            for (int dir = 0; dir < 8; dir++) {
                int nx = p.x + dx[dir];
                int ny = p.y + dy[dir];
                if (nx < 0 || nx >= (int)w || ny < 0 || ny >= (int)h) continue;
                size_t nIdx = (size_t)ny * w + (size_t)nx;
                if (visited[nIdx]) continue;

                float r = pixels[nIdx * 4 + 0];
                float g = pixels[nIdx * 4 + 1];
                float b = pixels[nIdx * 4 + 2];

                float stepDist = sqrtf((r - refR)*(r - refR) + (g - refG)*(g - refG) + (b - refB)*(b - refB));
                float globalDist = sqrtf((r - meanBgR)*(r - meanBgR) + (g - meanBgG)*(g - meanBgG) + (b - meanBgB)*(b - meanBgB));
                bool hasSlateChroma = ((b - g) >= 1.5f && (b - r) >= 2.5f && b >= 15.0f && b <= meanBgB + 16.0f);
                bool isSmoothSlate = hasSlateChroma && (stepDist < 8.5f) && (globalDist < 24.0f) && (grad[nIdx] < 11.0f);
                bool isSoftAmbientBg = (stepDist < 12.0f) && (grad[nIdx] < 16.0f) && ((r + g + b) >= 54.0f) && (r < 110.0f && g < 118.0f && b < 118.0f) && ((r + g + b) < 240.0f);
                bool isDropShadowBg = ((r + g + b) >= 42.0f && (r + g + b) <= 68.0f) &&
                                      ((b - r) >= 0.0f && (b - r) <= 7.0f) &&
                                      ((b - g) >= -1.5f && (b - g) <= 5.0f) &&
                                      (stepDist < 28.0f) && (grad[nIdx] < 24.0f);

                if (isSmoothSlate || isSoftAmbientBg || isDropShadowBg) {
                    visited[nIdx] = 1;
                    isBgRegion[nIdx] = 1;
                    bgR[nIdx] = refR * 0.85f + r * 0.15f;
                    bgG[nIdx] = refG * 0.85f + g * 0.15f;
                    bgB[nIdx] = refB * 0.85f + b * 0.15f;
                    queue[qTail++] = (Point2D){nx, ny};
                }
            }
        }

        // Clean up topological noise:
        // 1) Any interior background component (not touching image border) with < 220 pixels is reverted to foreground (prevents scale pinholes)
        // 2) Any foreground island (!isBgRegion) with < 220 pixels is converted to background (eliminates stray JPEG/dust specks in outer background)
        uint8_t *ccVis = (uint8_t *)calloc(w * h, 1);
        Point2D *ccBuf = (Point2D *)malloc(w * h * sizeof(Point2D));
        for (size_t i = 0; i < w * h; i++) {
            if (ccVis[i]) continue;
            uint8_t regionVal = isBgRegion[i];
            size_t cHead = 0, cTail = 0;
            ccVis[i] = 1;
            ccBuf[cTail++] = (Point2D){(int)(i % w), (int)(i / w)};
            bool touchesBorder = false;

            while (cHead < cTail) {
                Point2D p = ccBuf[cHead++];
                if (p.x == 0 || p.x == (int)w - 1 || p.y == 0 || p.y == (int)h - 1) {
                    touchesBorder = true;
                }
                for (int dir = 0; dir < 8; dir++) {
                    int nx = p.x + dx[dir];
                    int ny = p.y + dy[dir];
                    if (nx < 0 || nx >= (int)w || ny < 0 || ny >= (int)h) continue;
                    size_t nIdx = (size_t)ny * w + (size_t)nx;
                    if (!ccVis[nIdx] && isBgRegion[nIdx] == regionVal) {
                        ccVis[nIdx] = 1;
                        ccBuf[cTail++] = (Point2D){nx, ny};
                    }
                }
            }

            if (regionVal == 1 && !touchesBorder && cTail < 220) {
                // Small false background hole inside creature -> restore to foreground
                for (size_t k = 0; k < cTail; k++) {
                    size_t idx = (size_t)ccBuf[k].y * w + (size_t)ccBuf[k].x;
                    isBgRegion[idx] = 0;
                }
            } else if (regionVal == 0 && cTail < 220) {
                // Small stray dust/JPEG speck in background -> clear to background
                for (size_t k = 0; k < cTail; k++) {
                    size_t idx = (size_t)ccBuf[k].y * w + (size_t)ccBuf[k].x;
                    isBgRegion[idx] = 1;
                    bgR[idx] = meanBgR;
                    bgG[idx] = meanBgG;
                    bgB[idx] = meanBgB;
                }
            }
        }
        free(ccBuf);
        free(ccVis);

        // Compute distance from non-background creature pixels (distToFg) to restrict contact shadows
        int *distToFg = (int *)malloc(w * h * sizeof(int));
        qHead = 0;
        qTail = 0;
        for (size_t i = 0; i < w * h; i++) {
            if (!isBgRegion[i]) {
                distToFg[i] = 0;
                queue[qTail++] = (Point2D){(int)(i % w), (int)(i / w)};
            } else {
                distToFg[i] = 999999;
            }
        }
        while (qHead < qTail) {
            Point2D p = queue[qHead++];
            size_t curIdx = (size_t)p.y * w + (size_t)p.x;
            int d = distToFg[curIdx];
            if (d >= 12) continue;
            for (int dir = 0; dir < 8; dir++) {
                int nx = p.x + dx[dir];
                int ny = p.y + dy[dir];
                if (nx < 0 || nx >= (int)w || ny < 0 || ny >= (int)h) continue;
                size_t nIdx = (size_t)ny * w + (size_t)nx;
                if (distToFg[nIdx] > d + 1) {
                    distToFg[nIdx] = d + 1;
                    queue[qTail++] = (Point2D){nx, ny};
                }
            }
        }

        // Propagate local background color inward by up to 8 pixels into the creature/shadow boundary
        int *distToBg = (int *)malloc(w * h * sizeof(int));
        qHead = 0;
        qTail = 0;
        for (size_t i = 0; i < w * h; i++) {
            if (isBgRegion[i]) {
                distToBg[i] = 0;
                queue[qTail++] = (Point2D){(int)(i % w), (int)(i / w)};
            } else {
                distToBg[i] = 999999;
            }
        }
        while (qHead < qTail) {
            Point2D p = queue[qHead++];
            size_t curIdx = (size_t)p.y * w + (size_t)p.x;
            int d = distToBg[curIdx];
            if (d >= 8) continue;
            for (int dir = 0; dir < 8; dir++) {
                int nx = p.x + dx[dir];
                int ny = p.y + dy[dir];
                if (nx < 0 || nx >= (int)w || ny < 0 || ny >= (int)h) continue;
                size_t nIdx = (size_t)ny * w + (size_t)nx;
                if (distToBg[nIdx] > d + 1) {
                    distToBg[nIdx] = d + 1;
                    bgR[nIdx] = bgR[curIdx];
                    bgG[nIdx] = bgG[curIdx];
                    bgB[nIdx] = bgB[curIdx];
                    queue[qTail++] = (Point2D){nx, ny};
                }
            }
        }

        // Compute alpha and decontaminate edge colors
        for (size_t i = 0; i < w * h; i++) {
            float r = pixels[i * 4 + 0];
            float g = pixels[i * 4 + 1];
            float b = pixels[i * 4 + 2];
            float br = bgR[i] > 0 ? bgR[i] : meanBgR;
            float bg_ = bgG[i] > 0 ? bgG[i] : meanBgG;
            float bb = bgB[i] > 0 ? bgB[i] : meanBgB;

            if (isBgRegion[i]) {
                pixels[i * 4 + 0] = 0;
                pixels[i * 4 + 1] = 0;
                pixels[i * 4 + 2] = 0;
                pixels[i * 4 + 3] = 0;
            } else {
                int d = distToBg[i];
                if (d <= 3) {
                    // Boundary feathering & slate color decontamination
                    float cDist = sqrtf((r - br)*(r - br) + (g - bg_)*(g - bg_) + (b - bb)*(b - bb));
                    float greenWarmth = fmaxf(0.0f, (g - b) + 4.0f) + fmaxf(0.0f, (r - b) + 8.0f);
                    float bgLum = 0.2126f * br + 0.7152f * bg_ + 0.0722f * bb;
                    float pixLum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                    // If the pixel is much darker than the slate background, it's the dark outline/shadow of the creature -> keep high alpha!
                    float darkOutlineBoost = pixLum < bgLum - 5.0f ? clampf((bgLum - pixLum - 3.0f) / 14.0f, 0.0f, 1.0f) : 0.0f;
                    float colAlpha = clampf((cDist + greenWarmth * 1.5f - 3.5f) / 18.0f, 0.0f, 1.0f);
                    float geomAlpha = clampf((float)(d - 1) / 2.0f, 0.0f, 1.0f);
                    float alpha = clampf(fmaxf(darkOutlineBoost, fmaxf(colAlpha, geomAlpha * 0.90f)), 0.0f, 1.0f);

                    if (alpha > 0.05f && alpha < 0.99f && pixLum >= bgLum - 6.0f) {
                        r = clampf((r - (1.0f - alpha) * br) / alpha, 0.0f, 255.0f);
                        g = clampf((g - (1.0f - alpha) * bg_) / alpha, 0.0f, 255.0f);
                        b = clampf((b - (1.0f - alpha) * bb) / alpha, 0.0f, 255.0f);
                    }
                    pixels[i * 4 + 0] = (uint8_t)lrintf(r);
                    pixels[i * 4 + 1] = (uint8_t)lrintf(g);
                    pixels[i * 4 + 2] = (uint8_t)lrintf(b);
                    pixels[i * 4 + 3] = (uint8_t)lrintf(alpha * 255.0f);
                } else {
                    pixels[i * 4 + 3] = 255;
                }
            }
        }

        bool ok = saveRGBA(argv[2], pixels, w, h);

        free(distToBg);
        free(distToFg);
        free(grad);
        free(queue);
        free(isBgRegion);
        free(visited);
        free(bgB);
        free(bgG);
        free(bgR);
        free(pixels);

        return ok ? 0 : 1;
    }
}
