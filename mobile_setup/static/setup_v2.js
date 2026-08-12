(() => {
    "use strict";

    const INDOOR_OFFSET = 0.10;
    const OUTDOOR_OFFSET = 0.10;
    const MIN_LINE_LENGTH = 0.22;
    const HARD_EDGE_MARGIN = 0.04;
    const WARNING_EDGE_MARGIN = 0.08;
    const ROI_MARGIN = 0.06;

    const steps = [...document.querySelectorAll(".step")];
    const progressBar = document.getElementById("progressBar");
    const stepLabel = document.getElementById("stepLabel");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");

    const kakaoButton = document.getElementById("kakaoButton");
    const kakaoStatus = document.getElementById("kakaoStatus");

    const canvas = document.getElementById("boundaryCanvas");
    const ctx = canvas.getContext("2d");

    const resetLine = document.getElementById("resetLine");
    const reverseDirection = document.getElementById("reverseDirection");
    const finishButton = document.getElementById("finishButton");
    const finishStatus = document.getElementById("finishStatus");
    const validationMessage =
        document.getElementById("validationMessage");

    let currentStep = 1;
    let direction = 1;
    let dragging = null;

    let points = {
        a: {
            x: 0.22,
            y: 0.56
        },
        b: {
            x: 0.78,
            y: 0.56
        }
    };

    function showStep(step) {
        currentStep = Math.max(1, Math.min(3, step));

        steps.forEach((item) => {
            item.classList.toggle(
                "active",
                Number(item.dataset.step) === currentStep
            );
        });

        progressBar.style.width =
            `${(currentStep / 3) * 100}%`;

        stepLabel.textContent =
            `${currentStep} / 3 단계`;

        prevButton.disabled = currentStep === 1;

        nextButton.style.display =
            currentStep === 3 ? "none" : "block";

        if (currentStep === 3) {
            requestAnimationFrame(resizeCanvas);
        }

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }

    function clamp(value, minimum, maximum) {
        return Math.max(
            minimum,
            Math.min(maximum, value)
        );
    }

    function shiftLine(line, nx, ny, amount) {
        return line.map((point) => ({
            x: point.x + nx * amount,
            y: point.y + ny * amount
        }));
    }

    function outsideFrame(line) {
        return line.some((point) => (
            point.x < 0
            || point.x > 1
            || point.y < 0
            || point.y > 1
        ));
    }

    function nearEdge(line, margin) {
        return line.some((point) => (
            point.x < margin
            || point.x > 1 - margin
            || point.y < margin
            || point.y > 1 - margin
        ));
    }

    function buildGeometry() {
        const dx = points.b.x - points.a.x;
        const dy = points.b.y - points.a.y;
        const length = Math.hypot(dx, dy);

        const errors = [];
        const warnings = [];

        if (length < MIN_LINE_LENGTH) {
            errors.push(
                "현관 경계선이 너무 짧습니다. " +
                "사람이 통과하는 전체 폭을 포함하도록 선을 늘려 주세요."
            );
        }

        let nx = 0;
        let ny = 1;

        if (length > 0) {
            nx = (-dy / length) * direction;
            ny = (dx / length) * direction;
        }

        const boundaryLine = [
            {
                x: points.a.x,
                y: points.a.y
            },
            {
                x: points.b.x,
                y: points.b.y
            }
        ];

        const indoorLine = shiftLine(
            boundaryLine,
            nx,
            ny,
            -INDOOR_OFFSET
        );

        const outdoorLine = shiftLine(
            boundaryLine,
            nx,
            ny,
            OUTDOOR_OFFSET
        );

        if (outsideFrame(outdoorLine)) {
            errors.push(
                "외출 영역이 화면 밖으로 벗어납니다. " +
                "카메라 각도를 바깥쪽이 더 보이도록 조정하거나 " +
                "경계선을 화면 안쪽으로 옮겨 주세요."
            );
        } else if (nearEdge(outdoorLine, HARD_EDGE_MARGIN)) {
            errors.push(
                "현관 바깥쪽 감지 공간이 부족합니다. " +
                "카메라 각도를 바깥이 더 보이게 조정해 주세요."
            );
        } else if (nearEdge(outdoorLine, WARNING_EDGE_MARGIN)) {
            warnings.push(
                "외출 영역이 화면 가장자리에 가깝습니다. " +
                "바깥쪽 공간을 조금 더 확보하는 것을 권장합니다."
            );
        }

        if (outsideFrame(indoorLine)) {
            errors.push(
                "실내 감지 영역이 화면 밖으로 벗어납니다. " +
                "카메라가 실내 이동 경로도 보이도록 조정해 주세요."
            );
        } else if (nearEdge(indoorLine, HARD_EDGE_MARGIN)) {
            errors.push(
                "현관 안쪽 감지 공간이 부족합니다. " +
                "실내 이동 경로가 더 보이도록 카메라를 조정해 주세요."
            );
        }

        const allPoints = [
            ...indoorLine,
            ...boundaryLine,
            ...outdoorLine
        ];

        const minX = clamp(
            Math.min(...allPoints.map((point) => point.x)) - ROI_MARGIN,
            0,
            1
        );

        const minY = clamp(
            Math.min(...allPoints.map((point) => point.y)) - ROI_MARGIN,
            0,
            1
        );

        const maxX = clamp(
            Math.max(...allPoints.map((point) => point.x)) + ROI_MARGIN,
            0,
            1
        );

        const maxY = clamp(
            Math.max(...allPoints.map((point) => point.y)) + ROI_MARGIN,
            0,
            1
        );

        return {
            valid: errors.length === 0,
            errors,
            warnings,
            boundaryLine,
            indoorLine,
            outdoorLine,
            directionVector: {
                x: nx,
                y: ny
            },
            roi: {
                minX,
                minY,
                maxX,
                maxY
            }
        };
    }

    function updateValidation(geometry) {
        validationMessage.classList.remove(
            "ok",
            "warning",
            "error"
        );

        if (geometry.errors.length > 0) {
            validationMessage.classList.add("error");
            validationMessage.textContent =
                geometry.errors[0];
            finishButton.disabled = true;
            return;
        }

        if (geometry.warnings.length > 0) {
            validationMessage.classList.add("warning");
            validationMessage.textContent =
                geometry.warnings[0];
            finishButton.disabled = false;
            return;
        }

        validationMessage.classList.add("ok");
        validationMessage.textContent =
            "현재 설정을 사용할 수 있습니다.";
        finishButton.disabled = false;
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;

        canvas.width = Math.round(rect.width * ratio);
        canvas.height = Math.round(rect.height * ratio);

        ctx.setTransform(
            ratio,
            0,
            0,
            ratio,
            0,
            0
        );

        draw();
    }

    function screenPoint(point) {
        return {
            x: point.x * canvas.clientWidth,
            y: point.y * canvas.clientHeight
        };
    }

    function drawLine(
        line,
        color,
        width,
        dash = []
    ) {
        const start = screenPoint(line[0]);
        const end = screenPoint(line[1]);

        ctx.save();
        ctx.beginPath();
        ctx.setLineDash(dash);
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.restore();
    }

    function drawArrow(
        fromX,
        fromY,
        toX,
        toY,
        color,
        width
    ) {
        const angle = Math.atan2(
            toY - fromY,
            toX - fromX
        );

        const head = 14;

        ctx.save();

        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(toX, toY);

        ctx.lineTo(
            toX - head * Math.cos(angle - Math.PI / 6),
            toY - head * Math.sin(angle - Math.PI / 6)
        );

        ctx.lineTo(
            toX - head * Math.cos(angle + Math.PI / 6),
            toY - head * Math.sin(angle + Math.PI / 6)
        );

        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        ctx.restore();
    }

    function roundedRectangle(
        x,
        y,
        width,
        height,
        radius,
        fillStyle
    ) {
        ctx.save();
        ctx.beginPath();

        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);

        ctx.quadraticCurveTo(
            x + width,
            y,
            x + width,
            y + radius
        );

        ctx.lineTo(
            x + width,
            y + height - radius
        );

        ctx.quadraticCurveTo(
            x + width,
            y + height,
            x + width - radius,
            y + height
        );

        ctx.lineTo(x + radius, y + height);

        ctx.quadraticCurveTo(
            x,
            y + height,
            x,
            y + height - radius
        );

        ctx.lineTo(x, y + radius);

        ctx.quadraticCurveTo(
            x,
            y,
            x + radius,
            y
        );

        ctx.closePath();
        ctx.fillStyle = fillStyle;
        ctx.fill();
        ctx.restore();
    }

    function drawDirectionHint(
        midpoint,
        vector
    ) {
        const perpendicular = {
            x: -vector.y,
            y: vector.x
        };

        const scale = Math.min(
            canvas.clientWidth,
            canvas.clientHeight
        );

        const shaftStartDistance = scale * 0.04;
        const neckDistance = scale * 0.15;
        const tipDistance = scale * 0.25;
        const shaftHalfWidth = Math.max(8, scale * 0.025);
        const headHalfWidth = Math.max(23, scale * 0.070);

        const shaftStart = {
            x: midpoint.x + vector.x * shaftStartDistance,
            y: midpoint.y + vector.y * shaftStartDistance
        };

        const neck = {
            x: midpoint.x + vector.x * neckDistance,
            y: midpoint.y + vector.y * neckDistance
        };

        const tip = {
            x: midpoint.x + vector.x * tipDistance,
            y: midpoint.y + vector.y * tipDistance
        };

        ctx.save();

        ctx.shadowColor = "rgba(23, 54, 145, 0.35)";
        ctx.shadowBlur = 9;
        ctx.shadowOffsetY = 3;

        ctx.beginPath();

        ctx.moveTo(
            shaftStart.x + perpendicular.x * shaftHalfWidth,
            shaftStart.y + perpendicular.y * shaftHalfWidth
        );

        ctx.lineTo(
            neck.x + perpendicular.x * shaftHalfWidth,
            neck.y + perpendicular.y * shaftHalfWidth
        );

        ctx.lineTo(
            neck.x + perpendicular.x * headHalfWidth,
            neck.y + perpendicular.y * headHalfWidth
        );

        ctx.lineTo(
            tip.x,
            tip.y
        );

        ctx.lineTo(
            neck.x - perpendicular.x * headHalfWidth,
            neck.y - perpendicular.y * headHalfWidth
        );

        ctx.lineTo(
            neck.x - perpendicular.x * shaftHalfWidth,
            neck.y - perpendicular.y * shaftHalfWidth
        );

        ctx.lineTo(
            shaftStart.x - perpendicular.x * shaftHalfWidth,
            shaftStart.y - perpendicular.y * shaftHalfWidth
        );

        ctx.closePath();

        ctx.fillStyle = "rgba(72, 111, 255, 0.95)";
        ctx.fill();

        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(171, 193, 255, 0.95)";
        ctx.stroke();

        ctx.restore();

        const pillWidth = 132;
        const pillHeight = 42;

        const wantedLabelX =
            tip.x + vector.x * (pillHeight + 13);

        const wantedLabelY =
            tip.y + vector.y * (pillHeight + 13);

        const labelCenterX = clamp(
            wantedLabelX,
            pillWidth / 2 + 8,
            canvas.clientWidth - pillWidth / 2 - 8
        );

        const labelCenterY = clamp(
            wantedLabelY,
            pillHeight / 2 + 8,
            canvas.clientHeight - pillHeight / 2 - 8
        );

        roundedRectangle(
            labelCenterX - pillWidth / 2,
            labelCenterY - pillHeight / 2,
            pillWidth,
            pillHeight,
            16,
            "rgba(54, 84, 190, 0.78)"
        );

        ctx.save();
        ctx.font = "bold 18px sans-serif";
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        ctx.fillText(
            "\uC678\uCD9C \uBC29\uD5A5",
            labelCenterX,
            labelCenterY
        );

        ctx.restore();
    }

    function drawLabel(
        text,
        position,
        color
    ) {
        ctx.save();
        ctx.font = "bold 15px sans-serif";
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(
            text,
            position.x,
            position.y - 8
        );
        ctx.restore();
    }

    function draw() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const geometry = buildGeometry();

        ctx.clearRect(0, 0, width, height);

        const roiX = geometry.roi.minX * width;
        const roiY = geometry.roi.minY * height;
        const roiWidth =
            (geometry.roi.maxX - geometry.roi.minX) * width;
        const roiHeight =
            (geometry.roi.maxY - geometry.roi.minY) * height;

        ctx.save();
        ctx.setLineDash([8, 7]);
        ctx.strokeStyle = "rgba(173, 195, 255, 0.70)";
        ctx.lineWidth = 2;
        ctx.strokeRect(
            roiX,
            roiY,
            roiWidth,
            roiHeight
        );
        ctx.restore();

        drawLine(
            geometry.indoorLine,
            "#59c36a",
            4
        );

        drawLine(
            geometry.outdoorLine,
            geometry.errors.length > 0
                ? "#ff5048"
                : "#ef5b53",
            4,
            geometry.errors.length > 0
                ? [9, 6]
                : []
        );

        drawLine(
            geometry.boundaryLine,
            "#ffffff",
            5
        );

        const a = screenPoint(points.a);
        const b = screenPoint(points.b);

        [a, b].forEach((point) => {
            ctx.save();
            ctx.beginPath();
            ctx.arc(
                point.x,
                point.y,
                13,
                0,
                Math.PI * 2
            );
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#3865e8";
            ctx.lineWidth = 5;
            ctx.stroke();
            ctx.restore();
        });

        const indoorStart =
            screenPoint(geometry.indoorLine[0]);

        const boundaryStart =
            screenPoint(geometry.boundaryLine[0]);

        const outdoorStart =
            screenPoint(geometry.outdoorLine[0]);

        drawLabel(
            "실내 기준",
            indoorStart,
            "#67d47a"
        );

        drawLabel(
            "현관 경계",
            boundaryStart,
            "#ffffff"
        );

        drawLabel(
            "실외 기준",
            outdoorStart,
            "#ff7169"
        );

        const midpoint = {
            x: (a.x + b.x) / 2,
            y: (a.y + b.y) / 2
        };

        drawDirectionHint(
            midpoint,
            geometry.directionVector
        );

        updateValidation(geometry);
    }

    function pointerPosition(event) {
        const rect = canvas.getBoundingClientRect();

        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    canvas.addEventListener(
        "pointerdown",
        (event) => {
            const pointer = pointerPosition(event);
            const a = screenPoint(points.a);
            const b = screenPoint(points.b);

            const distanceA = Math.hypot(
                pointer.x - a.x,
                pointer.y - a.y
            );

            const distanceB = Math.hypot(
                pointer.x - b.x,
                pointer.y - b.y
            );

            dragging =
                distanceA <= distanceB
                    ? "a"
                    : "b";

            canvas.setPointerCapture(
                event.pointerId
            );
        }
    );

    canvas.addEventListener(
        "pointermove",
        (event) => {
            if (!dragging) {
                return;
            }

            const pointer =
                pointerPosition(event);

            points[dragging] = {
                x: clamp(
                    pointer.x / canvas.clientWidth,
                    0.02,
                    0.98
                ),
                y: clamp(
                    pointer.y / canvas.clientHeight,
                    0.02,
                    0.98
                )
            };

            draw();
        }
    );

    function stopDragging() {
        dragging = null;
    }

    canvas.addEventListener(
        "pointerup",
        stopDragging
    );

    canvas.addEventListener(
        "pointercancel",
        stopDragging
    );

    resetLine.addEventListener(
        "click",
        () => {
            points = {
                a: {
                    x: 0.22,
                    y: 0.56
                },
                b: {
                    x: 0.78,
                    y: 0.56
                }
            };

            direction = 1;
            finishStatus.textContent =
                "설정 대기 중";

            draw();
        }
    );

    reverseDirection.addEventListener(
        "click",
        () => {
            direction *= -1;
            finishStatus.textContent =
                "외출 방향을 변경했습니다.";
            draw();
        }
    );

    kakaoButton.addEventListener(
        "click",
        () => {
            kakaoStatus.textContent =
                "카카오 로그인 기능 연결 예정";
        }
    );

    prevButton.addEventListener(
        "click",
        () => {
            showStep(currentStep - 1);
        }
    );

    nextButton.addEventListener(
        "click",
        () => {
            showStep(currentStep + 1);
        }
    );

    finishButton.addEventListener(
        "click",
        async () => {
            const geometry = buildGeometry();

            updateValidation(geometry);

            if (!geometry.valid) {
                finishStatus.textContent =
                    "잘못된 설정을 먼저 수정해 주세요.";
                return;
            }

            finishButton.disabled = true;
            finishStatus.textContent =
                "설정을 저장하고 있습니다.";

            const payload = {
                boundary_line: {
                    a: {
                        x: points.a.x,
                        y: points.a.y
                    },
                    b: {
                        x: points.b.x,
                        y: points.b.y
                    }
                },
                direction
            };

            try {
                const saveResponse = await fetch(
                    "/api/setup/entrance",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body: JSON.stringify(payload)
                    }
                );

                const saveResult =
                    await saveResponse.json();

                if (!saveResponse.ok) {
                    throw new Error(
                        saveResult.errors?.[0]?.message
                        || saveResult.message
                        || "설정 저장에 실패했습니다."
                    );
                }

                finishStatus.textContent =
                    "설정 저장 완료. 감지 프로그램을 확인하고 있습니다.";

                const startResponse = await fetch(
                    "/api/system/start",
                    {
                        method: "POST"
                    }
                );

                const startResult =
                    await startResponse.json();

                if (
                    startResponse.status === 409
                    && startResult.code === "START_DISABLED"
                ) {
                    finishStatus.textContent =
                        "설정 저장 완료. 현재 노트북 테스트에서는 " +
                        "감지 자동 실행이 비활성화되어 있습니다.";

                    finishButton.disabled = false;
                    return;
                }

                if (!startResponse.ok) {
                    throw new Error(
                        startResult.message
                        || "감지 시작에 실패했습니다."
                    );
                }

                finishStatus.textContent =
                    startResult.message;

            } catch (error) {
                finishStatus.textContent =
                    error instanceof Error
                        ? error.message
                        : "처리 중 오류가 발생했습니다.";

                finishButton.disabled = false;
            }
        }
    );

    window.addEventListener(
        "resize",
        resizeCanvas
    );

    function refreshCameraPreview() {
        let image = null;

        if (currentStep === 2) {
            image = document.getElementById("cameraPreviewStep2");
        } else if (currentStep === 3) {
            image = document.getElementById("cameraPreviewStep3");
        }

        if (image) {
            image.src = "/camera/frame.jpg?t=" + Date.now();
        }
    }

    setInterval(refreshCameraPreview, 300);

    showStep(1);
})();
