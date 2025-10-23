# ── Stage 1: build environment ──
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget gnupg ca-certificates \
      ffmpeg \
      build-essential \
      libasound2-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -qO- https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

COPY . .

# ── Stage 2: production image ──
FROM python:3.11-slim

WORKDIR /app

# Install Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      libnss3 \
      libatk-bridge2.0-0 \
      libdrm2 \
      libxkbcommon0 \
      libxcomposite1 \
      libxdamage1 \
      libxrandr2 \
      libgbm1 \
      libxss1 \
      libasound2 \
      libatspi2.0-0 \
      libgtk-3-0 \
      libgdk-pixbuf-xlib-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy build artifacts
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create non-root user (security best practice)
RUN groupadd -r botuser && useradd -r -g botuser -m botuser \
    && chown -R botuser:botuser /app

# Switch to botuser
USER botuser

# Set runtime environment variables with defaults
ENV MEETING_URL="" \
    BOT_NAME="Clerk AI Bot" \
    PLATFORM="google_meet" \
    RT_GATEWAY_URL="ws://localhost:8001" \
    API_BASE_URL="http://localhost:8000" \
    JOIN_TIMEOUT_SEC=60 \
    AUDIO_SAMPLE_RATE=16000 \
    AUDIO_CHANNELS=1 \
    CHROME_BIN="/usr/bin/google-chrome" \
    CHROME_PATH="/usr/bin/google-chrome" \
    BROWSER_USE_API_KEY="" \
    LOG_LEVEL="info"

# Expose port for WebSocket / bot control
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py || exit 1

# Start the bot entry script
ENTRYPOINT ["python", "bot_entry.py"]
