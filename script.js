// Load real model output automatically
fetch("data.json")
  .then(response => response.json())
  .then(data => {
    // Use real data from JSON
    const dates = data.dates;
    const actual = data.actual;
    const predicted = data.predicted;
    const futureDates = data.futureDates;
    const forecast = data.forecast;

    // Plotly chart
    const trace1 = {
      x: dates,
      y: actual,
      mode: "lines",
      name: "Actual Price",
      line: { color: "royalblue", width: 3 }
    };

    const trace2 = {
      x: dates,
      y: predicted,
      mode: "lines",
      name: "Predicted Price",
      line: { color: "orange", width: 3 }
    };

    const trace3 = {
      x: futureDates,
      y: forecast,
      mode: "lines+markers",
      name: "Next 30 Days",
      line: { color: "lime", width: 3, dash: "dash" }
    };

    const layout = {
      paper_bgcolor: "#0E1117",
      plot_bgcolor: "#0E1117",
      font: { color: "#E0E0E0" },
      xaxis: { title: "Date" },
      yaxis: { title: "Price (USD)" },
      hovermode: "x unified",
    };

    Plotly.newPlot("chart", [trace1, trace2, trace3], layout);

    // Forecast Table
    const tbody = document.querySelector("#forecast tbody");
    futureDates.forEach((date, i) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${date}</td><td>$${forecast[i].toFixed(2)}</td>`;
      tbody.appendChild(row);
    });
  })
  .catch(error => {
    console.error("Error loading data.json:", error);
    alert("Could not load data.json. Please check the file path or server setup.");
  });
