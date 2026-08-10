"use strict";

const steps = Array.from(document.querySelectorAll(".step"));
const previousButton = document.getElementById("previousButton");
const nextButton = document.getElementById("nextButton");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

const canvas = document.getElementById("calibrationCanvas");
const context = canvas.getContext("2d");

let currentStep = 1;

const calibration = {
    roi: {
        x: 0.10,
        y: 0.10,
        width: 0.80,
        height: 0.80,
    },
    indoorLine: {
        start: { x: 0.20, y: 0.68 },
        end: { x: 0.80, y: 0.68 },
    },
    outdoorLine: {
        start: { x: 0.20, y: 0.38 },
        end: { x: 0.80, y: 0.38 },
    },
    direction: "indoor_to_outdoor",
};

function showStep(stepNumber) {
    currentStep = Math.max(1, Math.min(steps.length, stepNumber));

    steps.forEach((step) => {
        step.classList.toggle(
            "active",
            Number(step.dataset.step) === currentStep,
        );
    });

    const percent = (currentStep / steps.length) * 100;
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${currentStep} / ${steps.length} 단계`;

    previousButton.disabled = currentStep === 1;
    nextButton.textContent =
        currentStep === steps.length ? "설정 완료" : "다음";

    if (currentStep === 3) {
        requestAnimationFrame(resizeCalibrationCanvas);
    }
}

function resizeCalibrationCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;

    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawCalibration(rect.width, rect.height);
}

function drawCalibration(width, height) {
    context.clearRect(0, 0, width, height);

    context.fillStyle = "#0f172a";
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#cbd5e1";
    context.font = "14px Arial";
    context.textAlign = "center";
    context.fillText(
        "카메라 연결 후 실제 영상이 표시됩니다.",
        width / 2,
        height / 2,
    );

    const roiX = calibration.roi.x * width;
    const roiY = calibration.roi.y * height;
    const roiWidth = calibration.roi.width * width;
    const roiHeight = calibration.roi.height * height;

    context.strokeStyle = "#f8fafc";
    context.lineWidth = 2;
    context.setLineDash([8, 6]);
    context.strokeRect(roiX, roiY, roiWidth, roiHeight);
    context.setLineDash([]);

    drawLine(
        calibration.indoorLine,
        width,
        height,
        "#22c55e",
        "실내선",
    );

    drawLine(
        calibration.outdoorLine,
        width,
        height,
        "#ef4444",
        "실외선",
    );
}

function drawLine(line, width, height, color, label) {
    const startX = line.start.x * width;
    const startY = line.start.y * height;
    const endX = line.end.x * width;
    const endY = line.end.y * height;

    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 4;
    context.lineCap = "round";

    context.beginPath();
    context.moveTo(startX, startY);
    context.lineTo(endX, endY);
    context.stroke();

    for (const point of [
        { x: startX, y: startY },
        { x: endX, y: endY },
    ]) {
        context.beginPath();
        context.arc(point.x, point.y, 8, 0, Math.PI * 2);
        context.fill();
    }

    context.font = "bold 14px Arial";
    context.textAlign = "left";
    context.fillText(label, startX, startY - 12);
}

previousButton.addEventListener("click", () => {
    showStep(currentStep - 1);
});

nextButton.addEventListener("click", () => {
    if (currentStep < steps.length) {
        showStep(currentStep + 1);
        return;
    }

    document.getElementById("systemStatus").textContent =
        "설정 화면 준비 완료";
});

document
    .getElementById("resetCalibrationButton")
    .addEventListener("click", () => {
        calibration.roi = {
            x: 0.10,
            y: 0.10,
            width: 0.80,
            height: 0.80,
        };
        calibration.indoorLine = {
            start: { x: 0.20, y: 0.68 },
            end: { x: 0.80, y: 0.68 },
        };
        calibration.outdoorLine = {
            start: { x: 0.20, y: 0.38 },
            end: { x: 0.80, y: 0.38 },
        };
        resizeCalibrationCanvas();
    });

document
    .getElementById("reverseDirectionButton")
    .addEventListener("click", (event) => {
        calibration.direction =
            calibration.direction === "indoor_to_outdoor"
                ? "outdoor_to_indoor"
                : "indoor_to_outdoor";

        event.currentTarget.textContent =
            calibration.direction === "indoor_to_outdoor"
                ? "방향 반대로 설정"
                : "현재: 빨간선 → 초록선";
    });

window.addEventListener("resize", () => {
    if (currentStep === 3) {
        resizeCalibrationCanvas();
    }
});

showStep(1);