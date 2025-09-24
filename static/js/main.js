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
        }
        updateFileList();
        submitBtn.disabled = files.length === 0;
    }

    function updateFileList() {
        fileList.innerHTML = '';
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `<span><i class="fas fa-file-image"></i>${file.name}</span><button type="button" class="btn btn-link text-danger p-0" data-index="${index}" aria-label="Remove file"><i class="fas fa-times"></i></button>`;
            fileList.appendChild(fileItem);
        });
        document.querySelectorAll('.file-item button').forEach(button => {
            button.addEventListener('click', (e) => {
                files.splice(parseInt(e.currentTarget.dataset.index, 10), 1);
                updateFileList();
                submitBtn.disabled = files.length === 0;
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
            <h3>پردازش با موفقیت انجام شد</h3>
            <hr>
            <pre class="ocr-text">${decodedText}</pre>
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
});