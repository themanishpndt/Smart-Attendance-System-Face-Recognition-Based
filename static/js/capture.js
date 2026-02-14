// ========================================
// Web-Based Face Capture System
// Production-Ready Client-Side Logic
// ========================================

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    targetImages: 50,              // Reduced for faster processing
    uploadUrl: '/api/upload_capture',
    captureInterval: 1,            // Faster: 1 second interval
    imageQuality: 0.92,            // Higher quality for better accuracy
    minFaceDetections: 20,         // Minimum required
    confidenceThreshold: 0.5,      // Face detection confidence
  };

  // Model Ready Flag
  let faceApiReady = false;

  // State Management
  let state = {
    isCapturing: false,
    imageCount: 0,
    images: [],
    username: '',
    email: '',
    userId: '',
    department: '',
    phone: '',
    role: '',
    notes: '',
    stream: null,
    animationId: null,
    frameCount: 0,
    lastCaptureTime: 0,
  };

  // UI Elements
  const UI = {
    usernameForm: null,
    usernameInput: null,
    emailInput: null,
    userIdInput: null,
    departmentInput: null,
    phoneInput: null,
    roleInput: null,
    notesInput: null,
    proceedBtn: null,
    captureInterface: null,
    video: null,
    canvas: null,
    ctx: null,
    statusText: null,
    progressBar: null,
    progressText: null,
    startBtn: null,
    stopBtn: null,
    submitBtn: null,
    capturedCount: null,
    remainingCount: null,
  };

  /**
   * Load Face-API Models
   */
  async function loadFaceApiModels() {
    if (typeof faceapi === 'undefined') {
      console.warn('Face-API library not loaded, using interval-based capture');
      faceApiReady = false;
      return;
    }

    try {
      const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
      console.log('Loading Face-API models...');
      
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      
      faceApiReady = true;
      console.log('✓ Face-API TinyFaceDetector loaded successfully');
      showStatus('Face detection ready! Capturing images...', 'success');
    } catch (error) {
      console.warn('Face-API models not available, using interval-based capture:', error);
      faceApiReady = false;
      showStatus('Using basic capture mode (no face detection)', 'info');
    }
  }

  /**
   * Initialize UI Element References
   */
  function initializeUI() {
    UI.usernameForm = document.getElementById('username-form');
    UI.usernameInput = document.getElementById('capture-username');
    UI.emailInput = document.getElementById('capture-email');
    UI.userIdInput = document.getElementById('capture-id');
    UI.departmentInput = document.getElementById('capture-department');
    UI.phoneInput = document.getElementById('capture-phone');
    UI.roleInput = document.getElementById('capture-role');
    UI.notesInput = document.getElementById('capture-notes');
    UI.proceedBtn = document.getElementById('proceed-camera-btn');
    UI.captureInterface = document.getElementById('capture-interface');
    UI.video = document.getElementById('camera-feed');
    UI.canvas = document.getElementById('face-detection-overlay');
    UI.statusText = document.getElementById('status-text');
    UI.progressBar = document.getElementById('capture-progress');
    UI.progressText = document.getElementById('progress-text');
    UI.startBtn = document.getElementById('start-capture-btn');
    UI.stopBtn = document.getElementById('stop-capture-btn');
    UI.submitBtn = document.getElementById('submit-capture-btn');
    UI.capturedCount = document.getElementById('captured-count');
    UI.remainingCount = document.getElementById('remaining-count');

    if (UI.canvas) {
      UI.ctx = UI.canvas.getContext('2d');
    }

    // Set initial button states
    if (UI.startBtn) UI.startBtn.disabled = true;
    if (UI.stopBtn) UI.stopBtn.disabled = true;
    if (UI.submitBtn) UI.submitBtn.disabled = true;

    console.log('UI initialized:', {
      video: !!UI.video,
      canvas: !!UI.canvas,
      ctx: !!UI.ctx,
      startBtn: !!UI.startBtn,
      stopBtn: !!UI.stopBtn,
      submitBtn: !!UI.submitBtn
    });
  }

  /**
   * Bind Event Listeners
   */
  function bindEvents() {
    if (UI.proceedBtn) {
      UI.proceedBtn.addEventListener('click', handleProceedToCamera);
    }
    if (UI.startBtn) {
      UI.startBtn.addEventListener('click', startCapture);
    }
    if (UI.stopBtn) {
      UI.stopBtn.addEventListener('click', stopCapture);
    }
    if (UI.submitBtn) {
      UI.submitBtn.addEventListener('click', submitCaptures);
    }
  }

  /**
   * Handle "Proceed to Camera"
   */
  async function handleProceedToCamera() {
    const username = UI.usernameInput.value.trim();
    const email = UI.emailInput.value.trim();
    const userId = UI.userIdInput.value.trim();
    const department = UI.departmentInput.value;
    const phone = UI.phoneInput.value.trim();
    const role = UI.roleInput.value;
    const notes = UI.notesInput.value.trim();

    // Validate required fields
    if (!username || !/^[A-Za-z\s]+$/.test(username)) {
      showStatus('Please enter a valid name (letters and spaces only)', 'danger');
      UI.usernameInput.focus();
      return;
    }

    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      showStatus('Please enter a valid email address', 'danger');
      UI.emailInput.focus();
      return;
    }

    if (!userId) {
      showStatus('Please enter your ID number', 'danger');
      UI.userIdInput.focus();
      return;
    }

    if (!department) {
      showStatus('Please select your department', 'danger');
      UI.departmentInput.focus();
      return;
    }

    if (!role) {
      showStatus('Please select your role', 'danger');
      UI.roleInput.focus();
      return;
    }

    // Store all data in state
    state.username = username;
    state.email = email;
    state.userId = userId;
    state.department = department;
    state.phone = phone;
    state.role = role;
    state.notes = notes;

    console.log('Registration data:', { username, email, userId, department, role });

    if (UI.usernameForm) UI.usernameForm.style.display = 'none';
    if (UI.captureInterface) UI.captureInterface.style.display = 'block';

    showStatus('Requesting camera access...', 'info');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          aspectRatio: { ideal: 16/9 }
        },
        audio: false,
      });

      state.stream = stream;
      if (UI.video) {
        UI.video.srcObject = stream;
        console.log('Camera stream started');
      }

      showStatus('Camera ready! Click "Start Capture" to begin.', 'success');
      if (UI.startBtn) {
        UI.startBtn.disabled = false;
        console.log('Start button enabled');
      }
    } catch (error) {
      console.error('Camera error:', error);
      showStatus('Camera access denied. Please allow permissions.', 'danger');
      if (UI.usernameForm) UI.usernameForm.style.display = 'block';
      if (UI.captureInterface) UI.captureInterface.style.display = 'none';
    }
  }

  /**
   * Start Capture
   */
  function startCapture() {
    if (state.isCapturing) return;

    console.log('Starting capture...');

    state.isCapturing = true;
    state.imageCount = 0;
    state.images = [];
    state.frameCount = 0;
    state.lastCaptureTime = Date.now();

    if (UI.startBtn) UI.startBtn.disabled = true;
    if (UI.stopBtn) UI.stopBtn.disabled = false;
    showStatus('Loading face detection models...', 'info');

    // Add recording class to video wrapper
    const videoContainer = document.getElementById('video-container');
    if (videoContainer) videoContainer.classList.add('recording');

    // Load Face-API models before starting capture loop
    loadFaceApiModels().then(() => {
      console.log('Models loaded, starting capture loop');
      showStatus('Capturing images... Position your face in the center.', 'info');
      captureLoop();
    }).catch((error) => {
      console.warn('Failed to load Face-API models, starting basic capture:', error);
      showStatus('Capturing images (no face detection)', 'warning');
      captureLoop();
    });
  }

  /**
   * Capture Loop
   */
  function captureLoop() {
    if (!state.isCapturing) return;

    state.frameCount++;

    if (UI.canvas && UI.video && UI.ctx) {
      // Always draw current video frame to canvas
      UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);
      
      // Run face detection and capture
      detectFaces();
    }

    state.animationId = requestAnimationFrame(captureLoop);
  }

  /**
   * Detect Faces and Capture
   */
  async function detectFaces() {
    if (!UI.canvas || !UI.video) return;

    const now = Date.now();
    const timeSinceLastCapture = now - state.lastCaptureTime;

    try {
      // If Face-API is ready, use face detection
      if (faceApiReady && typeof faceapi !== 'undefined') {
        const detections = await faceapi.detectAllFaces(
          UI.video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 416,
            scoreThreshold: CONFIG.confidenceThreshold
          })
        );

        if (detections && detections.length > 0) {
          drawFaceBox(detections[0]);

          // Capture every N seconds when face is detected
          if (timeSinceLastCapture >= CONFIG.captureInterval * 1000) {
            captureImage();
            state.lastCaptureTime = now;
          }
        } else {
          // No face detected - show message
          drawNoFaceMessage();
        }
      } else {
        // Fallback: interval-based capture without face detection
        if (timeSinceLastCapture >= CONFIG.captureInterval * 1000) {
          captureImage();
          state.lastCaptureTime = now;
        }
      }
    } catch (error) {
      console.warn('Detection error:', error);
      // On error, still try to capture
      if (timeSinceLastCapture >= CONFIG.captureInterval * 1000) {
        captureImage();
        state.lastCaptureTime = now;
      }
    }
  }

  /**
   * Draw Face Bounding Box
   */
  function drawFaceBox(detection) {
    if (!UI.ctx || !UI.canvas || !UI.video) return;

    // Redraw video frame
    UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);

    // Get bounding box
    const box = detection.box || detection.detection?.box;

    UI.ctx.strokeStyle = '#00ff00';
    UI.ctx.lineWidth = 3;
    UI.ctx.strokeRect(box.x, box.y, box.width, box.height);

    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;

    UI.ctx.beginPath();
    UI.ctx.moveTo(centerX - 10, centerY);
    UI.ctx.lineTo(centerX + 10, centerY);
    UI.ctx.moveTo(centerX, centerY - 10);
    UI.ctx.lineTo(centerX, centerY + 10);
    UI.ctx.stroke();

    drawCenterGuide();
  }

  /**
   * Draw Center Guide
   */
  function drawCenterGuide() {
    if (!UI.ctx || !UI.canvas) return;

    const centerX = UI.canvas.width / 2;
    const centerY = UI.canvas.height / 2;

    UI.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    UI.ctx.lineWidth = 1;
    UI.ctx.setLineDash([5, 5]);

    // Horizontal line
    UI.ctx.beginPath();
    UI.ctx.moveTo(0, centerY);
    UI.ctx.lineTo(UI.canvas.width, centerY);
    UI.ctx.stroke();

    // Vertical line
    UI.ctx.beginPath();
    UI.ctx.moveTo(centerX, 0);
    UI.ctx.lineTo(centerX, UI.canvas.height);
    UI.ctx.stroke();

    UI.ctx.setLineDash([]);
  }

  /**
   * Draw No Face Detected Message
   */
  function drawNoFaceMessage() {
    if (!UI.ctx || !UI.canvas || !UI.video) return;

    // Redraw video frame
    UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);

    // Draw warning box in center
    const centerX = UI.canvas.width / 2;
    const centerY = UI.canvas.height / 2;

    UI.ctx.strokeStyle = '#ff0000';
    UI.ctx.lineWidth = 3;
    UI.ctx.strokeRect(centerX - 100, centerY - 75, 200, 150);

    UI.ctx.fillStyle = '#ff0000';
    UI.ctx.font = 'bold 16px Arial';
    UI.ctx.textAlign = 'center';
    UI.ctx.fillText('NO FACE DETECTED', centerX, centerY);
    UI.ctx.font = '14px Arial';
    UI.ctx.fillText('Position your face here', centerX, centerY + 25);
    UI.ctx.textAlign = 'left';
  }

  /**
   * Capture Image
   */
  function captureImage() {
    if (!UI.canvas || state.imageCount >= CONFIG.targetImages) return;

    try {
      const imageData = UI.canvas.toDataURL('image/jpeg', CONFIG.imageQuality);
      state.images.push(imageData);
      state.imageCount++;

      console.log(`Captured image ${state.imageCount}/${CONFIG.targetImages}`);

      updateProgress();
      playBeep();

      if (state.imageCount >= CONFIG.targetImages) {
        console.log('Target reached, stopping capture');
        stopCapture();
      }
    } catch (error) {
      console.error('Capture error:', error);
    }
  }

  /**
   * Update Progress
   */
  function updateProgress() {
    const percent = (state.imageCount / CONFIG.targetImages) * 100;

    if (UI.progressBar) {
      UI.progressBar.style.width = percent + '%';
      UI.progressBar.setAttribute('aria-valuenow', state.imageCount);
    }

    if (UI.progressText) {
      UI.progressText.textContent = `${state.imageCount} / ${CONFIG.targetImages}`;
    }

    if (UI.capturedCount) {
      UI.capturedCount.textContent = state.imageCount;
    }

    if (UI.remainingCount) {
      UI.remainingCount.textContent = Math.max(0, CONFIG.targetImages - state.imageCount);
    }
  }

  /**
   * Stop Capture
   */
  function stopCapture() {
    if (!state.isCapturing) return;

    state.isCapturing = false;

    if (state.animationId) {
      cancelAnimationFrame(state.animationId);
    }

    if (UI.ctx && UI.canvas) {
      UI.ctx.clearRect(0, 0, UI.canvas.width, UI.canvas.height);
    }

    // Remove recording class from video wrapper
    const videoContainer = document.getElementById('video-container');
    if (videoContainer) videoContainer.classList.remove('recording');

    if (UI.startBtn) UI.startBtn.disabled = false;
    if (UI.stopBtn) UI.stopBtn.disabled = true;
    if (UI.submitBtn) UI.submitBtn.disabled = state.imageCount < CONFIG.minFaceDetections;

    const message = `Captured ${state.imageCount} images. Click "Submit & Save" to register.`;
    showStatus(message, 'success');
  }

  /**
   * Submit Captures
   */
  async function submitCaptures() {
    if (state.imageCount < CONFIG.minFaceDetections) {
      showStatus(`Need at least ${CONFIG.minFaceDetections} images`, 'danger');
      return;
    }

    if (!state.username) {
      showStatus('Username is missing', 'danger');
      return;
    }

    // Show confirmation dialog
    const confirmed = await showConfirmDialog(
      `Ready to save ${state.imageCount} face images for ${state.username}?`,
      'This will train the recognition model.'
    );

    if (!confirmed) {
      showStatus('Submission cancelled. You can retake or submit again.', 'info');
      return;
    }

    if (UI.submitBtn) UI.submitBtn.disabled = true;
    showStatus('Uploading and processing images...', 'info');

    try {
      const response = await fetch(CONFIG.uploadUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: state.username,
          email: state.email,
          userId: state.userId,
          department: state.department,
          phone: state.phone,
          role: state.role,
          notes: state.notes,
          images: state.images,
        }),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        showStatus(`✓ ${result.message}`, 'success');
        // Show detailed success page
        setTimeout(() => {
          showSuccessDetails(result);
        }, 500);
      } else {
        showStatus(`Error: ${result.error || 'Upload failed'}`, 'danger');
        if (UI.submitBtn) UI.submitBtn.disabled = false;
      }
    } catch (error) {
      console.error('Submission error:', error);
      showStatus(`Network error: ${error.message}`, 'danger');
      if (UI.submitBtn) UI.submitBtn.disabled = false;
    }
  }

  /**
   * Show Success Details Page
   */
  function showSuccessDetails(result) {
    const backdrop = document.createElement('div');
    backdrop.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.95);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      animation: fadeIn 0.3s ease;
    `;

    const successCard = document.createElement('div');
    successCard.style.cssText = `
      background: white;
      border-radius: 20px;
      padding: 3rem;
      max-width: 600px;
      width: 90%;
      box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5);
      animation: scaleIn 0.4s ease;
    `;

    const imagesCount = state.imageCount;
    const username = state.username;
    const email = state.email;
    const userId = state.userId;
    const department = state.department;
    const role = state.role;

    successCard.innerHTML = `
      <style>
        @keyframes scaleIn {
          from { transform: scale(0.8); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
        @keyframes checkmark {
          0% { transform: scale(0) rotate(45deg); }
          50% { transform: scale(1.2) rotate(45deg); }
          100% { transform: scale(1) rotate(45deg); }
        }
        .checkmark-circle {
          width: 100px;
          height: 100px;
          background: linear-gradient(135deg, #06d6a0 0%, #00b894 100%);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 2rem;
          box-shadow: 0 10px 30px rgba(6, 214, 160, 0.3);
        }
        .checkmark {
          color: white;
          font-size: 3rem;
          animation: checkmark 0.5s ease 0.2s both;
        }
        .success-title {
          color: #1a1a2e;
          font-size: 2rem;
          font-weight: bold;
          text-align: center;
          margin-bottom: 1rem;
        }
        .success-subtitle {
          color: #666;
          text-align: center;
          margin-bottom: 2rem;
          font-size: 1.1rem;
        }
        .detail-box {
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-radius: 12px;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
          border-left: 4px solid #06d6a0;
        }
        .detail-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 0;
          border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        .detail-row:last-child {
          border-bottom: none;
        }
        .detail-label {
          color: #666;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .detail-value {
          color: #1a1a2e;
          font-weight: bold;
          font-size: 1.1rem;
        }
        .btn-home {
          width: 100%;
          padding: 1rem;
          background: linear-gradient(135deg, #06d6a0 0%, #00b894 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 1.1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s;
          box-shadow: 0 4px 15px rgba(6, 214, 160, 0.3);
        }
        .btn-home:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(6, 214, 160, 0.4);
        }
      </style>
      
      <div class="checkmark-circle">
        <i class="fas fa-check checkmark"></i>
      </div>
      
      <h2 class="success-title">Registration Successful!</h2>
      <p class="success-subtitle">Your face data has been captured and saved</p>
      
      <div class="detail-box">
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-user" style="color: #06d6a0;"></i>
            Full Name
          </span>
          <span class="detail-value">${username}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-envelope" style="color: #06d6a0;"></i>
            Email
          </span>
          <span class="detail-value" style="font-size: 0.95rem;">${email}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-id-card" style="color: #06d6a0;"></i>
            ID Number
          </span>
          <span class="detail-value">${userId}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-building" style="color: #06d6a0;"></i>
            Department
          </span>
          <span class="detail-value">${department}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-user-tag" style="color: #06d6a0;"></i>
            Role
          </span>
          <span class="detail-value">${role}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-camera" style="color: #06d6a0;"></i>
            Images Captured
          </span>
          <span class="detail-value">${imagesCount} Photos</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-database" style="color: #06d6a0;"></i>
            Dataset Status
          </span>
          <span class="detail-value" style="color: #06d6a0;">
            <i class="fas fa-check-circle"></i> Created
          </span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-brain" style="color: #06d6a0;"></i>
            Model Status
          </span>
          <span class="detail-value" style="color: #06d6a0;">
            <i class="fas fa-check-circle"></i> Trained
          </span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">
            <i class="fas fa-clock" style="color: #06d6a0;"></i>
            Registration Time
          </span>
          <span class="detail-value">${new Date().toLocaleTimeString()}</span>
        </div>
      </div>
      
      <div style="background: #e3f9f1; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; text-align: center;">
        <i class="fas fa-info-circle" style="color: #06d6a0; margin-right: 0.5rem;"></i>
        <span style="color: #00b894; font-weight: 600;">
          You can now use the attendance system!
        </span>
      </div>
      
      <button class="btn-home" onclick="window.location.href='/'">
        <i class="fas fa-home" style="margin-right: 0.5rem;"></i>
        Go to Home
      </button>
    `;

    backdrop.appendChild(successCard);
    document.body.appendChild(backdrop);
  }

  /**
   * Show Confirmation Dialog
   */
  function showConfirmDialog(title, message) {
    return new Promise((resolve) => {
      // Create modal backdrop
      const backdrop = document.createElement('div');
      backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 9998;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.2s ease;
      `;

      // Create modal
      const modal = document.createElement('div');
      modal.style.cssText = `
        background: white;
        border-radius: 16px;
        padding: 2rem;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        z-index: 9999;
        animation: slideUp 0.3s ease;
      `;

      modal.innerHTML = `
        <style>
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
          }
        </style>
        <div style="text-align: center;">
          <div style="font-size: 3rem; margin-bottom: 1rem;">
            <i class="fas fa-check-circle" style="color: #06d6a0;"></i>
          </div>
          <h3 style="color: #1a1a2e; margin-bottom: 0.5rem;">${title}</h3>
          <p style="color: #666; margin-bottom: 2rem;">${message}</p>
          <div style="display: flex; gap: 1rem; justify-content: center;">
            <button id="confirm-cancel" style="
              padding: 0.75rem 1.5rem;
              border: 2px solid #666;
              background: white;
              color: #666;
              border-radius: 8px;
              font-size: 1rem;
              font-weight: 600;
              cursor: pointer;
              transition: all 0.3s;
            ">
              <i class="fas fa-times mr-2"></i>Cancel
            </button>
            <button id="confirm-retake" style="
              padding: 0.75rem 1.5rem;
              border: 2px solid #ff6b6b;
              background: white;
              color: #ff6b6b;
              border-radius: 8px;
              font-size: 1rem;
              font-weight: 600;
              cursor: pointer;
              transition: all 0.3s;
            ">
              <i class="fas fa-redo mr-2"></i>Retake
            </button>
            <button id="confirm-submit" style="
              padding: 0.75rem 1.5rem;
              border: none;
              background: linear-gradient(135deg, #06d6a0 0%, #00b894 100%);
              color: white;
              border-radius: 8px;
              font-size: 1rem;
              font-weight: 600;
              cursor: pointer;
              box-shadow: 0 4px 15px rgba(6, 214, 160, 0.3);
              transition: all 0.3s;
            ">
              <i class="fas fa-save mr-2"></i>Submit & Save
            </button>
          </div>
        </div>
      `;

      backdrop.appendChild(modal);
      document.body.appendChild(backdrop);

      // Handle button clicks
      const closeDialog = (result) => {
        backdrop.style.animation = 'fadeIn 0.2s ease reverse';
        setTimeout(() => {
          backdrop.remove();
          resolve(result);
        }, 200);
      };

      modal.querySelector('#confirm-cancel').onclick = () => closeDialog(false);
      modal.querySelector('#confirm-retake').onclick = () => {
        closeDialog(false);
        // Reset and restart capture
        setTimeout(() => {
          state.isCapturing = false;
          state.imageCount = 0;
          state.images = [];
          state.frameCount = 0;
          state.lastCaptureTime = Date.now();
          if (UI.submitBtn) UI.submitBtn.disabled = true;
          if (UI.startBtn) UI.startBtn.disabled = false;
          updateProgress();
          showStatus('Ready to retake. Click "Start Capture" when ready.', 'info');
        }, 300);
      };
      modal.querySelector('#confirm-submit').onclick = () => closeDialog(true);

      // Add hover effects
      const buttons = modal.querySelectorAll('button');
      buttons.forEach(btn => {
        btn.onmouseenter = () => btn.style.transform = 'translateY(-2px)';
        btn.onmouseleave = () => btn.style.transform = 'translateY(0)';
      });
    });
  }

  /**
   * Show Status Message
   */
  function showStatus(message, type = 'info') {
    if (UI.statusText) {
      UI.statusText.textContent = message;
    }

    const statusEl = document.getElementById('capture-status');
    if (statusEl) {
      const icons = {
        success: 'check-circle',
        danger: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle',
      };
      const icon = icons[type] || 'info-circle';
      statusEl.className = `alert alert-${type} mb-4`;
      statusEl.innerHTML = `<i class="fas fa-${icon} mr-2"></i>${message}`;
    }
  }

  /**
   * Play Beep Sound
   */
  function playBeep() {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();

      osc.connect(gain);
      gain.connect(audioContext.destination);

      osc.frequency.value = 800;
      osc.type = 'sine';

      gain.gain.setValueAtTime(0.3, audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

      osc.start(audioContext.currentTime);
      osc.stop(audioContext.currentTime + 0.1);
    } catch (e) {
      console.debug('Audio not available');
    }
  }

  /**
   * Initialize on Page Load
   */
  document.addEventListener('DOMContentLoaded', function() {
    initializeUI();
    bindEvents();
    console.log('✓ Face Capture System Ready');
  });

  /**
   * Cleanup on Page Unload
   */
  window.addEventListener('beforeunload', function() {
    if (state.stream) {
      state.stream.getTracks().forEach(track => track.stop());
    }
    if (state.animationId) {
      cancelAnimationFrame(state.animationId);
    }
  });
})();
