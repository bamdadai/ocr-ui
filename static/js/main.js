// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. DOM Element Selection ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const resultsContainer = document.getElementById('results-container');
    const loadingOverlay = document.querySelector('.loading-overlay');
    let files = [];
    let pollingIntervalId = null;
    let currentUploadedFile = null; // Store the uploaded file for preview

    // --- 2. Event Handlers ---
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => dropZone.addEventListener(eventName, preventDefaults, false));
    ['dragenter', 'dragover'].forEach(eventName => dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight')));
    ['dragleave', 'drop'].forEach(eventName => dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight')));
    dropZone.addEventListener('drop', (e) => handleFiles([...e.dataTransfer.files]));
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        handleFiles([...e.target.files]);
        e.target.value = '';
    });

    // --- 3. File Management ---
    function handleFiles(newFiles) {
        if (newFiles.length > 0) {
            files = [newFiles[0]];
            // Add success animation to upload area
            dropZone.style.borderColor = 'var(--success-color)';
            dropZone.style.backgroundColor = 'rgba(72, 187, 120, 0.1)';
            setTimeout(() => {
                dropZone.style.borderColor = '';
                dropZone.style.backgroundColor = '';
            }, 1000);
        }
        updateFileList();
        submitBtn.disabled = files.length === 0;
    }

    function updateFileList() {
        fileList.innerHTML = '';
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.style.opacity = '0';
            fileItem.style.transform = 'translateY(10px)';
            
            // Get file size in a readable format
            let fileSize = (file.size / 1024).toFixed(1) + ' KB';
            if (file.size >= 1024 * 1024) {
                fileSize = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
            }
            
            fileItem.innerHTML = `
                <span>
                    <i class="fas fa-file-image"></i>
                    <strong>${file.name}</strong>
                    <small class="text-muted d-block">${fileSize}</small>
                </span>
                <button type="button" class="btn btn-link text-danger p-1" data-index="${index}" aria-label="Remove file">
                    <i class="fas fa-times"></i>
                </button>
            `;
            fileList.appendChild(fileItem);
            
            // Animate in
            setTimeout(() => {
                fileItem.style.transition = 'all 0.3s ease';
                fileItem.style.opacity = '1';
                fileItem.style.transform = 'translateY(0)';
            }, 50);
        });
        
        document.querySelectorAll('.file-item button').forEach(button => {
            button.addEventListener('click', (e) => {
                const fileItem = e.currentTarget.closest('.file-item');
                fileItem.style.transform = 'translateX(-100%)';
                fileItem.style.opacity = '0';
                setTimeout(() => {
                    files.splice(parseInt(e.currentTarget.dataset.index, 10), 1);
                    updateFileList();
                    submitBtn.disabled = files.length === 0;
                }, 300);
            });
        });
    }

    // --- 4. Form Submission ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (files.length === 0) return;
        if (pollingIntervalId) clearInterval(pollingIntervalId);

        loadingOverlay.classList.remove('d-none');
        resultsContainer.innerHTML = '';

        // Store the uploaded file for preview
        currentUploadedFile = files[0];
        
        const formData = new FormData();
        formData.append('file', files[0]);
        formData.append('guid', self.crypto.randomUUID());
        
        try {
            const response = await fetch('/v3/ocr', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) {
                const errorDetail = data.detail || 'خطای ناشناخته از سمت سرور';
                throw new Error(typeof errorDetail === 'object' ? JSON.stringify(errorDetail) : errorDetail);
            }
            const mainTaskID = data.task_id;
            if (mainTaskID) {
                pollForTaskResult(mainTaskID);
            } else {
                throw new Error("سرور شناسه تسک معتبری برنگرداند.");
            }
        } catch (error) {
            displayError(`خطا در ارسال: ${error.message}`);
            loadingOverlay.classList.add('d-none');
        }
    });
    
    // --- 5. Polling and Display Functions ---

    /**
     * Decodes a Base64 string that contains UTF-8 characters.
     * @param {string} base64 - The Base64 encoded string.
     * @returns {string} The decoded UTF-8 string.
     */
    function decodeBase64Utf8(base64) {
        try {
            const binaryString = atob(base64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return new TextDecoder().decode(bytes);
        } catch (e) {
            console.error("Failed to decode Base64 string:", e);
            return "خطا در کدگشایی متن.";
        }
    }

    function pollForTaskResult(taskId) {
        pollingIntervalId = setInterval(async () => {
            try {
                const response = await fetch(`/v3/ocr/tasks/${taskId}`);
                if (!response.ok) throw new Error(`سرور با کد وضعیت ${response.status} پاسخ داد`);
                
                const data = await response.json();

                if (data.status === 'SUCCESS') {
                    clearInterval(pollingIntervalId);
                    loadingOverlay.classList.add('d-none');
                    displayFinalResult(data.result);
                } else if (data.status === 'FAILURE') {
                    clearInterval(pollingIntervalId);
                    loadingOverlay.classList.add('d-none');
                    const errorDetails = data.result?.error || 'پردازش تسک در سرور با خطا مواجه شد.';
                    displayError(errorDetails);
                }
            } catch (error) {
                clearInterval(pollingIntervalId);
                loadingOverlay.classList.add('d-none');
                displayError(`خطا در بررسی وضعیت تسک: ${error.message}`);
            }
        }, 3000);
    }

    function displayFinalResult(result) {
        resultsContainer.innerHTML = '';
        const resultCard = document.createElement('div');
        resultCard.className = 'alert alert-success';
        
        // Use the new, robust decoding function
        const decodedText = decodeBase64Utf8(result.text);

        resultCard.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h3 class="mb-0">
                    <i class="fas fa-check-circle text-success ms-2"></i>
                    پردازش با موفقیت انجام شد
                </h3>
                <button type="button" class="btn btn-outline-primary" onclick="previewOriginalImage()">
                    <i class="fas fa-image ms-2"></i>
                    مشاهده تصویر اصلی
                </button>
            </div>
            <div class="result-text-container">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0 text-muted">
                        <i class="fas fa-file-text ms-2"></i>
                        متن استخراج شده
                    </h5>
                    <button type="button" class="btn btn-sm btn-outline-secondary" onclick="copyToClipboard()">
                        <i class="fas fa-copy ms-1"></i>
                        کپی متن
                    </button>
                </div>
                <pre class="ocr-text" id="extracted-text">${decodedText}</pre>
            </div>
        `;
        resultsContainer.appendChild(resultCard);
    }
    

    function displayError(message) {
        resultsContainer.innerHTML = '';
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger';
        errorDiv.textContent = message;
        resultsContainer.appendChild(errorDiv);
    }

    // --- 6. Image Preview Functions ---
    window.previewOriginalImage = function() {
        if (!currentUploadedFile) {
            alert('فایل اصلی برای نمایش موجود نیست.');
            return;
        }

        // Check if it's an image file
        if (!currentUploadedFile.type.startsWith('image/')) {
            alert('فایل آپلود شده یک تصویر نیست.');
            return;
        }

        // Create and show modal
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'imagePreviewModal';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">تصویر اصلی</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img id="previewImage" class="img-fluid" style="max-height: 70vh;" alt="تصویر اصلی">
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">بستن</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Load the image
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImage').src = e.target.result;
        };
        reader.readAsDataURL(currentUploadedFile);
        
        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Clean up modal after closing
        modal.addEventListener('hidden.bs.modal', function() {
            document.body.removeChild(modal);
        });
    }

    // Copy to clipboard function
    window.copyToClipboard = function() {
        const textElement = document.getElementById('extracted-text');
        if (!textElement) {
            alert('متنی برای کپی کردن یافت نشد.');
            return;
        }

        const text = textElement.textContent;
        
        // Use modern clipboard API if available
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(() => {
                showCopySuccess();
            }).catch(() => {
                fallbackCopyTextToClipboard(text);
            });
        } else {
            fallbackCopyTextToClipboard(text);
        }
    }

    function fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showCopySuccess();
        } catch (err) {
            alert('خطا در کپی کردن متن.');
        }
        
        document.body.removeChild(textArea);
    }

    function showCopySuccess() {
        // Show temporary success message
        const copyBtn = document.querySelector('button[onclick="copyToClipboard()"]');
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check ms-1"></i>کپی شد!';
            copyBtn.classList.remove('btn-outline-secondary');
            copyBtn.classList.add('btn-success');
            
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove('btn-success');
                copyBtn.classList.add('btn-outline-secondary');
            }, 2000);
        }
    }
});