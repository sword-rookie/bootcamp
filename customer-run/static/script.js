const form = document.getElementById("predictionForm");

const predictionText = document.getElementById("predictionText");
const probabilityText = document.getElementById("probability");
const progressFill = document.getElementById("progressFill");
const loader = document.getElementById("loader");

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    loader.classList.remove("hidden");

    predictionText.innerHTML = "Predicting...";
    probabilityText.innerHTML = "--";
    progressFill.style.width = "0%";

    const data = {

        gender: document.getElementById("gender").value,
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),

        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,

        tenure: parseInt(document.getElementById("tenure").value),

        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,

        InternetService: document.getElementById("InternetService").value,

        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,

        DeviceProtection: document.getElementById("DeviceProtection").value,

        TechSupport: document.getElementById("TechSupport").value,

        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,

        Contract: document.getElementById("Contract").value,

        PaperlessBilling: document.getElementById("PaperlessBilling").value,

        PaymentMethod: document.getElementById("PaymentMethod").value,

        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),

        TotalCharges: parseFloat(document.getElementById("TotalCharges").value)

    };

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });

        const result = await response.json();

        loader.classList.add("hidden");

        const probability = Number(result.probability);

        const percent = Math.round(probability * 100);

        predictionText.innerHTML = result.prediction;

        probabilityText.innerHTML = percent + "%";

        progressFill.style.width = percent + "%";

        if (result.prediction === "Yes") {

            predictionText.style.color = "#FF6B6B";

            progressFill.style.background =
                "linear-gradient(90deg,#FF6B6B,#ff3b3b)";

        }

        else {

            predictionText.style.color = "#10B981";

            progressFill.style.background =
                "linear-gradient(90deg,#00F0FF,#10B981)";

        }

    }

    catch (error) {

        loader.classList.add("hidden");

        predictionText.innerHTML = "Error";

        probabilityText.innerHTML = "--";

        progressFill.style.width = "0%";

        console.error(error);

        alert("Unable to connect to the server.");

    }

});