/**
 * Browser Bot Entry Point
 * 
 * This script launches a headless browser to join meetings and stream audio
 * to the RT Gateway service for real-time processing.
 */

const { chromium } = require('playwright');
const WebSocket = require('ws');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const path = require('path');

// Simple logging configuration
const logger = {
  info: (msg, ...args) => console.log(`[INFO] ${msg}`, ...args),
  error: (msg, ...args) => console.error(`[ERROR] ${msg}`, ...args),
  warn: (msg, ...args) => console.warn(`[WARN] ${msg}`, ...args),
  debug: (msg, ...args) => {
    if (process.env.LOG_LEVEL === 'debug') {
      console.log(`[DEBUG] ${msg}`, ...args);
    }
  }
};

/**
 * Log events to the backend API
 */
async function logToBackend(event, data) {
  try {
    const apiUrl = process.env.API_BASE_URL || 'http://localhost:8000';
    const response = await axios.post(`${apiUrl}/api/v1/meetings/bot-log`, {
      event,
      data,
      timestamp: new Date().toISOString()
    }, {
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    logger.info('Logged to backend API', { event, status: response.status });
  } catch (error) {
    logger.error('Failed to log to backend API', { event, error: error.message });
  }
}

// Environment variables
const config = {
  meetingUrl: process.env.MEETING_URL,
  botName: process.env.BOT_NAME || 'Clerk AI Bot',
  platform: process.env.PLATFORM || 'google_meet',
  rtGatewayUrl: process.env.RT_GATEWAY_URL || 'ws://localhost:8001',
  apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:8000',
  joinTimeoutSec: parseInt(process.env.JOIN_TIMEOUT_SEC) || 60,
  audioSampleRate: parseInt(process.env.AUDIO_SAMPLE_RATE) || 16000,
  audioChannels: parseInt(process.env.AUDIO_CHANNELS) || 1,
  meetingId: process.env.MEETING_ID || uuidv4(),
  sessionId: uuidv4()
};

// Global state
let browser = null;
let page = null;
let wsConnection = null;
let audioStream = null;
let isJoined = false;
let isLeaving = false;

/**
 * Main entry point
 */
async function main() {
  try {
    logger.info('Starting Browser Bot', {
      meetingUrl: config.meetingUrl,
      botName: config.botName,
      platform: config.platform,
      sessionId: config.sessionId
    });

    // Validate required environment variables
    if (!config.meetingUrl) {
      throw new Error('MEETING_URL environment variable is required');
    }

    // Initialize WebSocket connection to RT Gateway
    await initializeWebSocketConnection();

    // Launch browser and join meeting
    await launchBrowserAndJoinMeeting();

  // Start audio streaming (optional - don't fail if not available)
  try {
    await startAudioStreaming();
  } catch (error) {
    logger.warn('Audio streaming not available, continuing without it', { error: error.message });
  }

    // Keep the bot running until meeting ends or interrupted
    await keepBotRunning();

  } catch (error) {
    logger.error('Browser Bot failed', { error: error.message, stack: error.stack });
    await cleanup();
    process.exit(1);
  }
}

/**
 * Initialize WebSocket connection to RT Gateway
 */
async function initializeWebSocketConnection() {
  return new Promise((resolve, reject) => {
    logger.info('Connecting to RT Gateway', { url: config.rtGatewayUrl });

    wsConnection = new WebSocket(config.rtGatewayUrl);

    wsConnection.on('open', () => {
      logger.info('Connected to RT Gateway');
      
      // Send bot registration message
      const registrationMessage = {
        type: 'bot_registration',
        sessionId: config.sessionId,
        meetingId: config.meetingId,
        botName: config.botName,
        platform: config.platform,
        audioConfig: {
          sampleRate: config.audioSampleRate,
          channels: config.audioChannels
        }
      };

      wsConnection.send(JSON.stringify(registrationMessage));
      resolve();
    });

    wsConnection.on('message', handleWebSocketMessage);
    wsConnection.on('error', (error) => {
      logger.error('WebSocket error', { error: error.message });
      reject(error);
    });

    wsConnection.on('close', () => {
      logger.warn('WebSocket connection closed');
    });
  });
}

/**
 * Handle incoming WebSocket messages
 */
function handleWebSocketMessage(data) {
  try {
    const message = JSON.parse(data.toString());
    
    switch (message.type) {
      case 'audio_request':
        handleAudioRequest(message);
        break;
      case 'tts_audio':
        handleTTSAudio(message);
        break;
      case 'meeting_command':
        handleMeetingCommand(message);
        break;
      default:
        logger.debug('Unknown message type', { type: message.type });
    }
  } catch (error) {
    logger.error('Error handling WebSocket message', { error: error.message });
  }
}

/**
 * Handle audio request from RT Gateway
 */
function handleAudioRequest(message) {
  logger.debug('Received audio request', { requestId: message.requestId });
  
  // Send current audio stream data if available
  if (audioStream && audioStream.isActive) {
    // This would be implemented based on the specific audio capture method
    // For now, we'll send a placeholder response
    const response = {
      type: 'audio_response',
      requestId: message.requestId,
      audioData: null, // Would contain actual audio data
      timestamp: Date.now()
    };
    
    wsConnection.send(JSON.stringify(response));
  }
}

/**
 * Handle TTS audio to be played in meeting
 */
async function handleTTSAudio(message) {
  logger.info('Received TTS audio to play', { audioId: message.audioId });
  
  try {
    // Inject audio into the meeting
    await injectAudioIntoMeeting(message.audioData);
    
    // Send confirmation
    const response = {
      type: 'tts_played',
      audioId: message.audioId,
      timestamp: Date.now()
    };
    
    wsConnection.send(JSON.stringify(response));
  } catch (error) {
    logger.error('Error playing TTS audio', { error: error.message });
  }
}

/**
 * Handle meeting commands (mute/unmute, etc.)
 */
async function handleMeetingCommand(message) {
  logger.info('Received meeting command', { command: message.command });
  
  try {
    switch (message.command) {
      case 'mute':
        await muteMicrophone();
        break;
      case 'unmute':
        await unmuteMicrophone();
        break;
      case 'leave':
        await leaveMeeting();
        break;
      default:
        logger.warn('Unknown command', { command: message.command });
    }
  } catch (error) {
    logger.error('Error handling meeting command', { error: error.message });
  }
}

/**
 * Launch browser and join meeting
 */
async function launchBrowserAndJoinMeeting() {
  logger.info('Launching browser...');

  // Launch browser with required arguments
  browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--disable-web-security',
      '--disable-features=VizDisplayCompositor',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--autoplay-policy=no-user-gesture-required',
      '--allow-running-insecure-content',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--disable-field-trial-config',
      '--disable-back-forward-cache',
      '--disable-ipc-flooding-protection',
      '--enable-features=NetworkService,NetworkServiceLogging',
      '--force-color-profile=srgb',
      '--metrics-recording-only',
      '--no-first-run',
      '--enable-automation',
      '--password-store=basic',
      '--use-mock-keychain'
    ]
  });

  // Create context with realistic headers and media constraints
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    permissions: ['microphone', 'camera'],
    viewport: { width: 1920, height: 1080 },
    extraHTTPHeaders: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Cache-Control': 'no-cache',
      'Pragma': 'no-cache',
      'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
      'Sec-Ch-Ua-Mobile': '?0',
      'Sec-Ch-Ua-Platform': '"macOS"',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
    },
    // Override media constraints to ensure camera/mic are available
    media: {
      video: {
        width: 640,
        height: 480,
        frameRate: 30
      },
      audio: {
        sampleRate: 44100,
        channels: 2
      }
    }
  });

  // Create new page from context
  page = await context.newPage();

  // Inject script to force enable camera/microphone
  await page.addInitScript(() => {
    // Override getUserMedia to always return mock streams
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia;
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      console.log('Bot: Intercepting getUserMedia call', constraints);
      
      // Create mock video stream
      if (constraints.video) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        
        // Draw a simple pattern to simulate video
        const drawFrame = () => {
          ctx.fillStyle = '#4CAF50';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'white';
          ctx.font = '24px Arial';
          ctx.textAlign = 'center';
          ctx.fillText('Clerk AI Bot', canvas.width/2, canvas.height/2);
          ctx.fillText('Active', canvas.width/2, canvas.height/2 + 30);
        };
        
        drawFrame();
        setInterval(drawFrame, 100);
        
        const stream = canvas.captureStream(30);
        return stream;
      }
      
      // Fallback to original if no video needed
      return originalGetUserMedia.call(navigator.mediaDevices, constraints);
    };
  });

  // Navigate to meeting URL with human-like behavior
  logger.info('Navigating to meeting URL', { url: config.meetingUrl });
  
  // Add some human-like delays and mouse movements
  await page.goto(config.meetingUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  
  // Simulate human behavior - move mouse around
  await page.mouse.move(100, 100);
  await page.waitForTimeout(1000);
  await page.mouse.move(200, 200);
  await page.waitForTimeout(1000);
  
  // Scroll a bit to simulate human behavior
  await page.evaluate(() => window.scrollTo(0, 100));
  await page.waitForTimeout(500);

  // Log page HTML for debugging
  const pageHTML = await page.content();
  logger.info('Page HTML loaded', { 
    title: await page.title(),
    url: page.url(),
    htmlLength: pageHTML.length,
    htmlPreview: pageHTML.substring(0, 1000) + '...'
  });

  // Join the meeting based on platform
  await joinMeetingByPlatform();

  // Wait for meeting to be fully loaded
  await waitForMeetingToLoad();

  isJoined = true;
  logger.info('Successfully joined meeting');
  
  // Log to backend API (optional - don't fail if not available)
  try {
    await logToBackend('meeting_joined', {
      meeting_id: config.meetingId,
      session_id: config.sessionId,
      platform: config.platform,
      bot_name: config.botName,
      meeting_url: config.meetingUrl,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.warn('Backend logging not available, continuing without it', { error: error.message });
  }
}

/**
 * Join meeting based on platform
 */
async function joinMeetingByPlatform() {
  logger.info('Joining meeting', { platform: config.platform });

  switch (config.platform) {
    case 'google_meet':
      await joinGoogleMeet();
      break;
    case 'zoom':
      await joinZoom();
      break;
    case 'teams':
      await joinTeams();
      break;
    default:
      throw new Error(`Unsupported platform: ${config.platform}`);
  }
}

/**
 * Join Google Meet
 */
async function joinGoogleMeet() {
  try {
    // Wait for join button and click it using JavaScript (bypasses visibility checks)
    const joinNowButton = await page.locator('[data-call-started="true"], [jsname="Qx7uuf"]').first();
    await joinNowButton.evaluate(element => element.click());
    logger.info('Clicked Google Meet join button (JavaScript click)');

    // Wait for meeting to load and permissions to be requested
    await page.waitForTimeout(8000);
    
    logger.info('Looking for camera/microphone controls...');
    
    // Try to enable camera and microphone more aggressively
    const cameraSelectors = [
      // Primary camera button selectors
      '[jsname="BOHaEe"]', // Google Meet camera button
      '[aria-label*="Turn on camera"]',
      '[aria-label*="Turn off camera"]',
      '[data-is-muted="true"][aria-label*="camera"]',
      '[data-is-muted="false"][aria-label*="camera"]',
      'button[aria-label*="camera"]',
      '[data-tooltip*="camera"]',
      // Alternative selectors
      '[aria-label*="Camera"]',
      'button[data-is-muted="true"]',
      'button[data-is-muted="false"]',
      // Generic button selectors
      'button[aria-label*="Turn on"]',
      'button[aria-label*="Turn off"]'
    ];
    
    let cameraFound = false;
    let microphoneFound = false;
    
    // Try to find and click camera button
    for (const selector of cameraSelectors) {
      try {
        const elements = await page.locator(selector).all();
        for (const element of elements) {
          const isVisible = await element.isVisible();
          const ariaLabel = await element.getAttribute('aria-label') || '';
          const textContent = await element.textContent() || '';
          
          logger.debug(`Checking element: ${selector}, visible: ${isVisible}, aria-label: ${ariaLabel}, text: ${textContent}`);
          
          if (isVisible && (ariaLabel.toLowerCase().includes('camera') || textContent.toLowerCase().includes('camera'))) {
            await element.click();
            logger.info(`Turned ON camera using selector: ${selector}`);
            cameraFound = true;
            await page.waitForTimeout(1000);
            break;
          }
        }
        if (cameraFound) break;
      } catch (error) {
        logger.debug(`Selector ${selector} error: ${error.message}`);
      }
    }
    
    // Try to find and click microphone button
    const micSelectors = [
      '[jsname="BOHaEe"]', // Google Meet mic button
      '[aria-label*="Turn on microphone"]',
      '[aria-label*="Turn off microphone"]',
      '[data-is-muted="true"][aria-label*="microphone"]',
      '[data-is-muted="false"][aria-label*="microphone"]',
      'button[aria-label*="microphone"]',
      '[data-tooltip*="microphone"]',
      '[aria-label*="Microphone"]',
      'button[aria-label*="mic"]'
    ];
    
    for (const selector of micSelectors) {
      try {
        const elements = await page.locator(selector).all();
        for (const element of elements) {
          const isVisible = await element.isVisible();
          const ariaLabel = await element.getAttribute('aria-label') || '';
          const textContent = await element.textContent() || '';
          
          if (isVisible && (ariaLabel.toLowerCase().includes('microphone') || ariaLabel.toLowerCase().includes('mic') || textContent.toLowerCase().includes('mic'))) {
            await element.click();
            logger.info(`Turned ON microphone using selector: ${selector}`);
            microphoneFound = true;
            await page.waitForTimeout(1000);
            break;
          }
        }
        if (microphoneFound) break;
      } catch (error) {
        logger.debug(`Mic selector ${selector} error: ${error.message}`);
      }
    }
    
    if (!cameraFound && !microphoneFound) {
      logger.warn('No camera/microphone controls found - bot will join without video/audio');
      logger.warn('Bot may not be visible in participant list');
    } else {
      logger.info(`Bot joined with camera: ${cameraFound}, microphone: ${microphoneFound}`);
    }
    
    // Wait a bit more for the meeting to fully initialize
    await page.waitForTimeout(3000);
    
    // Try to find and interact with any visible buttons after joining
    logger.info('Looking for any interactive elements after joining...');
    try {
      const allButtons = await page.locator('button').all();
      logger.info(`Found ${allButtons.length} buttons on the page`);
      
      for (let i = 0; i < Math.min(allButtons.length, 10); i++) {
        try {
          const button = allButtons[i];
          const isVisible = await button.isVisible();
          const ariaLabel = await button.getAttribute('aria-label') || '';
          const textContent = await button.textContent() || '';
          const className = await button.getAttribute('class') || '';
          
          logger.debug(`Button ${i}: visible=${isVisible}, aria-label="${ariaLabel}", text="${textContent}", class="${className}"`);
          
          // Look for camera/mic related buttons
          if (isVisible && (ariaLabel.toLowerCase().includes('camera') || 
                          ariaLabel.toLowerCase().includes('microphone') ||
                          ariaLabel.toLowerCase().includes('mic') ||
                          textContent.toLowerCase().includes('camera') ||
                          textContent.toLowerCase().includes('mic'))) {
            logger.info(`Found potential camera/mic button: ${ariaLabel || textContent}`);
            await button.click();
            logger.info('Clicked potential camera/mic button');
            await page.waitForTimeout(1000);
          }
        } catch (error) {
          logger.debug(`Error checking button ${i}: ${error.message}`);
        }
      }
    } catch (error) {
      logger.error('Error scanning for buttons', { error: error.message });
    }
    
  } catch (error) {
    logger.error('Error joining Google Meet', { error: error.message });
    throw error;
  }
}

/**
 * Join Zoom meeting
 */
async function joinZoom() {
  try {
    logger.info('Attempting to join Zoom meeting');
    
    // Log current page state for debugging
    const currentHTML = await page.content();
    logger.info('Zoom page HTML', { 
      title: await page.title(),
      url: page.url(),
      htmlLength: currentHTML.length,
      htmlPreview: currentHTML.substring(0, 2000) + '...'
    });

    // Wait for join button and click it
    const joinButton = await page.waitForSelector('#joinBtn, .join-btn', { timeout: 10000 });
    if (joinButton) {
      await joinButton.click();
      logger.info('Clicked Zoom join button');
    }

    // Handle name input
    const nameInput = await page.waitForSelector('#inputname', { timeout: 5000 });
    if (nameInput) {
      await nameInput.fill(config.botName);
      logger.info('Entered bot name');
    }

    // Click continue
    const continueButton = await page.waitForSelector('#joinBtn', { timeout: 5000 });
    if (continueButton) {
      await continueButton.click();
      logger.info('Clicked continue button');
    }

    // Turn off camera and microphone
    await page.waitForTimeout(2000);
    
    const cameraButton = await page.locator('[aria-label*="Stop Video"]').first();
    if (await cameraButton.isVisible()) {
      await cameraButton.click();
      logger.info('Turned off camera');
    }

    const micButton = await page.locator('[aria-label*="Mute"]').first();
    if (await micButton.isVisible()) {
      await micButton.click();
      logger.info('Turned off microphone');
    }

  } catch (error) {
    logger.error('Error joining Zoom meeting', { error: error.message });
    throw error;
  }
}

/**
 * Join Microsoft Teams meeting
 */
async function joinTeams() {
  try {
    // Wait for join button and click it
    const joinButton = await page.waitForSelector('[data-tid="prejoin-join-button"], .join-btn', { timeout: 10000 });
    if (joinButton) {
      await joinButton.click();
      logger.info('Clicked Teams join button');
    }

    // Turn off camera and microphone
    await page.waitForTimeout(2000);
    
    const cameraButton = await page.locator('[data-tid="toggle-camera"]').first();
    if (await cameraButton.isVisible()) {
      await cameraButton.click();
      logger.info('Turned off camera');
    }

    const micButton = await page.locator('[data-tid="toggle-mute"]').first();
    if (await micButton.isVisible()) {
      await micButton.click();
      logger.info('Turned off microphone');
    }

  } catch (error) {
    logger.error('Error joining Teams meeting', { error: error.message });
    throw error;
  }
}

/**
 * Wait for meeting to fully load
 */
async function waitForMeetingToLoad() {
  logger.info('Waiting for meeting to load...');
  
  // Wait for meeting controls to appear
  await page.waitForTimeout(5000);
  
  // Check if we're in the meeting by looking for common meeting UI elements
  const meetingIndicators = [
    '[data-call-started="true"]', // Google Meet
    '.meeting-client-view', // Zoom
    '[data-tid="meeting-page"]' // Teams
  ];

  for (const selector of meetingIndicators) {
    try {
      await page.waitForSelector(selector, { timeout: 10000 });
      logger.info('Meeting loaded successfully');
      return;
    } catch (error) {
      // Continue to next selector
    }
  }

  logger.warn('Meeting may not have loaded properly, continuing anyway');
}

/**
 * Start audio streaming to RT Gateway
 */
async function startAudioStreaming() {
  logger.info('Starting audio streaming...');

  try {
    // Inject audio capture script into the page
    await page.addInitScript(() => {
      // This script runs in the browser context
      window.audioContext = null;
      window.mediaStream = null;
      window.audioProcessor = null;

      // Function to start audio capture
      window.startAudioCapture = async function() {
        try {
          // Get user media (microphone)
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              sampleRate: 16000,
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true
            }
          });

          window.mediaStream = stream;
          window.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
          });

          const source = window.audioContext.createMediaStreamSource(stream);
          const processor = window.audioContext.createScriptProcessor(4096, 1, 1);

          processor.onaudioprocess = function(event) {
            const inputBuffer = event.inputBuffer;
            const inputData = inputBuffer.getChannelData(0);
            
            // Send audio data to parent process
            window.postMessage({
              type: 'audio_data',
              data: Array.from(inputData)
            }, '*');
          };

          source.connect(processor);
          processor.connect(window.audioContext.destination);
          window.audioProcessor = processor;

          return true;
        } catch (error) {
          console.error('Error starting audio capture:', error);
          return false;
        }
      };

      // Function to stop audio capture
      window.stopAudioCapture = function() {
        if (window.audioProcessor) {
          window.audioProcessor.disconnect();
          window.audioProcessor = null;
        }
        if (window.mediaStream) {
          window.mediaStream.getTracks().forEach(track => track.stop());
          window.mediaStream = null;
        }
        if (window.audioContext) {
          window.audioContext.close();
          window.audioContext = null;
        }
      };
    });

    // Start audio capture
    const audioStarted = await page.evaluate(() => window.startAudioCapture());
    if (!audioStarted) {
      throw new Error('Failed to start audio capture');
    }

    // Listen for audio data from the page
    page.on('console', (msg) => {
      if (msg.type() === 'log' && msg.text().includes('audio_data')) {
        // Handle audio data
        handleAudioData(msg.text());
      }
    });

    // Set up message listener for audio data
    page.on('pageerror', (error) => {
      logger.error('Page error', { error: error.message });
    });

    audioStream = {
      isActive: true,
      startTime: Date.now()
    };

    logger.info('Audio streaming started successfully');

  } catch (error) {
    logger.error('Error starting audio streaming', { error: error.message });
    throw error;
  }
}

/**
 * Handle audio data from the browser
 */
function handleAudioData(audioDataString) {
  try {
    // Parse audio data
    const audioData = JSON.parse(audioDataString);
    
    // Send to RT Gateway
    const message = {
      type: 'audio_stream',
      sessionId: config.sessionId,
      meetingId: config.meetingId,
      audioData: audioData.data,
      timestamp: Date.now()
    };

    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
      wsConnection.send(JSON.stringify(message));
    }
  } catch (error) {
    logger.error('Error handling audio data', { error: error.message });
  }
}

/**
 * Inject audio into the meeting
 */
async function injectAudioIntoMeeting(audioData) {
  try {
    logger.info('Injecting audio into meeting');

    // This would involve playing audio through the browser's audio system
    // Implementation depends on the specific platform and browser capabilities
    
    await page.evaluate((audioData) => {
      // Create audio context and play the audio
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = audioContext.createBuffer(1, audioData.length, 16000);
      audioBuffer.copyToChannel(new Float32Array(audioData), 0);
      
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start();
    }, audioData);

    logger.info('Audio injected successfully');

  } catch (error) {
    logger.error('Error injecting audio', { error: error.message });
    throw error;
  }
}

/**
 * Mute microphone
 */
async function muteMicrophone() {
  try {
    logger.info('Muting microphone');
    
    // Platform-specific mute implementation
    switch (config.platform) {
      case 'google_meet':
        const micButton = await page.locator('[data-is-muted="false"][aria-label*="microphone"]').first();
        if (await micButton.isVisible()) {
          await micButton.click();
        }
        break;
      case 'zoom':
        const zoomMicButton = await page.locator('[aria-label*="Unmute"]').first();
        if (await zoomMicButton.isVisible()) {
          await zoomMicButton.click();
        }
        break;
      case 'teams':
        const teamsMicButton = await page.locator('[data-tid="toggle-mute"]').first();
        if (await teamsMicButton.isVisible()) {
          await teamsMicButton.click();
        }
        break;
    }
    
    logger.info('Microphone muted');
  } catch (error) {
    logger.error('Error muting microphone', { error: error.message });
  }
}

/**
 * Unmute microphone
 */
async function unmuteMicrophone() {
  try {
    logger.info('Unmuting microphone');
    
    // Platform-specific unmute implementation
    switch (config.platform) {
      case 'google_meet':
        const micButton = await page.locator('[data-is-muted="true"][aria-label*="microphone"]').first();
        if (await micButton.isVisible()) {
          await micButton.click();
        }
        break;
      case 'zoom':
        const zoomMicButton = await page.locator('[aria-label*="Mute"]').first();
        if (await zoomMicButton.isVisible()) {
          await zoomMicButton.click();
        }
        break;
      case 'teams':
        const teamsMicButton = await page.locator('[data-tid="toggle-mute"]').first();
        if (await teamsMicButton.isVisible()) {
          await teamsMicButton.click();
        }
        break;
    }
    
    logger.info('Microphone unmuted');
  } catch (error) {
    logger.error('Error unmuting microphone', { error: error.message });
  }
}

/**
 * Leave the meeting
 */
async function leaveMeeting() {
  if (isLeaving) {
    return;
  }
  
  isLeaving = true;
  logger.info('Leaving meeting');

  try {
    // Platform-specific leave implementation
    switch (config.platform) {
      case 'google_meet':
        const leaveButton = await page.locator('[aria-label*="Leave call"]').first();
        if (await leaveButton.isVisible()) {
          await leaveButton.click();
        }
        break;
      case 'zoom':
        const zoomLeaveButton = await page.locator('[aria-label*="Leave"]').first();
        if (await zoomLeaveButton.isVisible()) {
          await zoomLeaveButton.click();
        }
        break;
      case 'teams':
        const teamsLeaveButton = await page.locator('[data-tid="leave-call-button"]').first();
        if (await teamsLeaveButton.isVisible()) {
          await teamsLeaveButton.click();
        }
        break;
    }

    // Send leave notification to API
    await notifyMeetingLeft();

    logger.info('Successfully left meeting');
  } catch (error) {
    logger.error('Error leaving meeting', { error: error.message });
  }
}

/**
 * Keep the bot running until meeting ends or interrupted
 */
async function keepBotRunning() {
  logger.info('Bot is running, waiting for meeting to end...');

  // Set up signal handlers for graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Received SIGINT, shutting down gracefully...');
    await cleanup();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('Received SIGTERM, shutting down gracefully...');
    await cleanup();
    process.exit(0);
  });

  // Keep running until interrupted or meeting ends
  while (!isLeaving) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Check if meeting is still active
    if (page) {
      try {
        const isInMeeting = await page.evaluate(() => {
          // Check for meeting indicators
          return document.querySelector('[data-call-started="true"], .meeting-client-view, [data-tid="meeting-page"]') !== null;
        });
        
        if (!isInMeeting) {
          logger.info('Meeting appears to have ended');
          break;
        }
      } catch (error) {
        logger.error('Error checking meeting status', { error: error.message });
        break;
      }
    }
  }

  await cleanup();
}

/**
 * Notify API that bot joined meeting
 */
async function notifyMeetingJoined() {
  try {
    const response = await axios.post(`${config.apiBaseUrl}/api/v1/meetings/${config.meetingId}/bot-joined`, {
      sessionId: config.sessionId,
      botName: config.botName,
      platform: config.platform,
      timestamp: new Date().toISOString()
    });

    logger.info('Notified API of meeting join', { status: response.status });
  } catch (error) {
    logger.error('Error notifying API of meeting join', { error: error.message });
  }
}

/**
 * Notify API that bot left meeting
 */
async function notifyMeetingLeft() {
  try {
    const response = await axios.post(`${config.apiBaseUrl}/api/v1/meetings/${config.meetingId}/bot-left`, {
      sessionId: config.sessionId,
      timestamp: new Date().toISOString()
    });

    logger.info('Notified API of meeting leave', { status: response.status });
  } catch (error) {
    logger.error('Error notifying API of meeting leave', { error: error.message });
  }
}

/**
 * Cleanup resources
 */
async function cleanup() {
  logger.info('Cleaning up resources...');

  try {
    // Stop audio streaming
    if (audioStream && audioStream.isActive) {
      if (page) {
        await page.evaluate(() => {
          if (window.stopAudioCapture) {
            window.stopAudioCapture();
          }
        });
      }
      audioStream.isActive = false;
    }

    // Close WebSocket connection
    if (wsConnection) {
      wsConnection.close();
      wsConnection = null;
    }

    // Close browser
    if (browser) {
      await browser.close();
      browser = null;
    }

    logger.info('Cleanup completed');
  } catch (error) {
    logger.error('Error during cleanup', { error: error.message });
  }
}

// Start the bot
if (require.main === module) {
  main().catch((error) => {
    logger.error('Fatal error', { error: error.message, stack: error.stack });
    process.exit(1);
  });
}

module.exports = {
  main,
  cleanup,
  config
};
