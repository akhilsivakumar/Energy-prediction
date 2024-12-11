function predict() {
    var month = document.getElementById("month").value;
    var year = document.getElementById("year").value;

    fetchPrediction(month, year);
}

function fetchPrediction(month, year) {
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ month: month, year: year })
    })
    .then(response => response.json())
    .then(data => {
        displayResult(data.prediction);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function displayResult(prediction) {
    var resultContainer = document.getElementById("result-container");
    resultContainer.innerHTML = `<p>Predicted electricity consumption: ${prediction}</p>`;
}
