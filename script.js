document.getElementById("churnForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(this);
    let data = {};
    formData.forEach((value, key) => {
        if (!isNaN(value) && value.trim() !== "") {
            data[key] = parseFloat(value); // convert numbers
        } else {
            data[key] = value;
        }
    });

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        document.getElementById("result").textContent =
            result.churn_prediction === 1
                ? "🚨 Customer is likely to churn"
                : "✅ Customer is not likely to churn";
    } catch (error) {
        console.error("Error:", error);
    }
});
