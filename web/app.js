// Neural Speech Denoising Studio - Client Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabPresetsBtn = document.getElementById("tabPresetsBtn");
  const tabUploadBtn = document.getElementById("tabUploadBtn");
  const presetsContainer = document.getElementById("presetsContainer");
  const uploadContainer = document.getElementById("uploadContainer");

  const presetCards = document.querySelectorAll(".preset-card");
  const snrSlider = document.getElementById("snrSlider");
  const snrDisplay = document.getElementById("snrDisplay");

  const audioDropzone = document.getElementById("audioDropzone");
  const fileInput = document.getElementById("fileInput");
  const uploadFileInfo = document.getElementById("uploadFileInfo");
  const uploadFileName = document.getElementById("uploadFileName");
  const clearFileBtn = document.getElementById("clearFileBtn");

  const processBtn = document.getElementById("processBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnText = document.getElementById("btnText");

  // Audio elements
  const audioClean = document.getElementById("audioClean");
  const audioNoisy = document.getElementById("audioNoisy");
  const audioDenoised = document.getElementById("audioDenoised");
  const cleanAudioCard = document.getElementById("cleanAudioCard");

  // A/B Toggle
  const abToggleBtn = document.getElementById("abToggleBtn");
  const abIndicator = document.getElementById("abIndicator");
  let abCurrent = "noisy"; // 'noisy' or 'denoised'

  // Telemetry
  const valNoisySnr = document.getElementById("valNoisySnr");
  const valDenoisedSnr = document.getElementById("valDenoisedSnr");
  const valSnrGain = document.getElementById("valSnrGain");
  const valSiSdr = document.getElementById("valSiSdr");

  // Spectrograms
  const imgSpecNoisy = document.getElementById("imgSpecNoisy");
  const imgSpecMask = document.getElementById("imgSpecMask");
  const imgSpecDenoised = document.getElementById("imgSpecDenoised");
  const placeholderNoisy = document.getElementById("placeholderNoisy");
  const placeholderMask = document.getElementById("placeholderMask");
  const placeholderDenoised = document.getElementById("placeholderDenoised");

  // State
  let activeTab = "presets"; // 'presets' | 'upload'
  let selectedPreset = "traffic";
  let uploadedFile = null;

  // Tab switching
  tabPresetsBtn.addEventListener("click", () => {
    activeTab = "presets";
    tabPresetsBtn.classList.add("active");
    tabUploadBtn.classList.remove("active");
    presetsContainer.classList.add("active");
    uploadContainer.classList.remove("active");
  });

  tabUploadBtn.addEventListener("click", () => {
    activeTab = "upload";
    tabUploadBtn.classList.add("active");
    tabPresetsBtn.classList.remove("active");
    uploadContainer.classList.add("active");
    presetsContainer.classList.remove("active");
  });

  // Preset Selection
  presetCards.forEach((card) => {
    card.addEventListener("click", () => {
      presetCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      selectedPreset = card.getAttribute("data-preset");
    });
  });

  // SNR Slider
  snrSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    snrDisplay.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(1)} dB`;
  });

  // File Upload Dropzone
  audioDropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  audioDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    audioDropzone.classList.add("dragover");
  });

  audioDropzone.addEventListener("dragleave", () => {
    audioDropzone.classList.remove("dragover");
  });

  audioDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    audioDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  clearFileBtn.addEventListener("click", () => {
    uploadedFile = null;
    fileInput.value = "";
    uploadFileInfo.classList.add("hidden");
    audioDropzone.classList.remove("hidden");
  });

  function handleFileSelected(file) {
    uploadedFile = file;
    uploadFileName.textContent = file.name;
    uploadFileInfo.classList.remove("hidden");
    audioDropzone.classList.add("hidden");
  }

  // Execution Trigger
  processBtn.addEventListener("click", async () => {
    setLoading(true);

    try {
      let response;
      if (activeTab === "presets") {
        const formData = new FormData();
        formData.append("preset_id", selectedPreset);
        formData.append("snr_db", snrSlider.value);
        response = await fetch("/api/denoise_preset", {
          method: "POST",
          body: formData,
        });
      } else {
        if (!uploadedFile) {
          alert("Please select or drop an audio file first!");
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append("audio_file", uploadedFile);
        response = await fetch("/api/denoise_upload", {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Inference failed");
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      console.error(err);
      alert(`Error during denoising: ${err.message}`);
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    if (isLoading) {
      processBtn.disabled = true;
      btnSpinner.classList.remove("hidden");
      btnText.textContent = "Processing Spectrograms & Denoising...";
    } else {
      processBtn.disabled = false;
      btnSpinner.classList.add("hidden");
      btnText.textContent = "Execute Neural Denoising";
    }
  }

  function displayResults(data) {
    // 1. Audio Sources
    if (data.audio_urls.clean) {
      cleanAudioCard.classList.remove("hidden");
      audioClean.src = data.audio_urls.clean;
    } else {
      cleanAudioCard.classList.add("hidden");
    }

    audioNoisy.src = data.audio_urls.noisy;
    audioDenoised.src = data.audio_urls.denoised;

    // 2. Spectrograms
    if (data.spectrograms.noisy) {
      imgSpecNoisy.src = `data:image/png;base64,${data.spectrograms.noisy}`;
      imgSpecNoisy.classList.remove("hidden");
      placeholderNoisy.classList.add("hidden");
    }
    if (data.spectrograms.mask) {
      imgSpecMask.src = `data:image/png;base64,${data.spectrograms.mask}`;
      imgSpecMask.classList.remove("hidden");
      placeholderMask.classList.add("hidden");
    }
    if (data.spectrograms.denoised) {
      imgSpecDenoised.src = `data:image/png;base64,${data.spectrograms.denoised}`;
      imgSpecDenoised.classList.remove("hidden");
      placeholderDenoised.classList.add("hidden");
    }

    // 3. Telemetry Metrics
    if (data.has_clean_reference) {
      valNoisySnr.textContent = `${data.metrics.noisy_snr > 0 ? "+" : ""}${data.metrics.noisy_snr} dB`;
      valDenoisedSnr.textContent = `+${data.metrics.denoised_snr} dB`;
      valSnrGain.textContent = `+${data.metrics.snr_gain} dB`;
      const sdrGain = (data.metrics.denoised_si_sdr - data.metrics.noisy_si_sdr).toFixed(2);
      valSiSdr.textContent = `+${sdrGain} dB`;
    } else {
      valNoisySnr.textContent = "User Audio";
      valDenoisedSnr.textContent = "Filtered";
      valSnrGain.textContent = "Applied";
      valSiSdr.textContent = "--";
    }

    // Auto-play Denoised output for immediate satisfaction
    setTimeout(() => {
      audioDenoised.play().catch(() => {});
      abCurrent = "denoised";
      updateAbIndicator();
    }, 300);
  }

  // A/B Comparison Switcher
  abToggleBtn.addEventListener("click", () => {
    toggleAB();
  });

  function toggleAB() {
    let activePlayer = abCurrent === "noisy" ? audioNoisy : audioDenoised;
    let targetPlayer = abCurrent === "noisy" ? audioDenoised : audioNoisy;

    const currentTime = activePlayer.currentTime || 0;
    const isPlaying = !activePlayer.paused;

    activePlayer.pause();
    targetPlayer.currentTime = currentTime;

    if (isPlaying) {
      targetPlayer.play().catch(() => {});
    }

    abCurrent = abCurrent === "noisy" ? "denoised" : "noisy";
    updateAbIndicator();
  }

  function updateAbIndicator() {
    if (abCurrent === "denoised") {
      abIndicator.textContent = "Denoised Output (B)";
      abIndicator.classList.add("denoised-active");
    } else {
      abIndicator.textContent = "Noisy Input (A)";
      abIndicator.classList.remove("denoised-active");
    }
  }

  // Trigger default preset on load for instant preview
  setTimeout(() => {
    processBtn.click();
  }, 400);
});
