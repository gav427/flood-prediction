document.addEventListener('DOMContentLoaded', function () {
    // API key
    const apiKey = '46307c0733115117bdbc3aa38ad0028a';

    // URL for forecast
    const forecastUrl = `https://api.openweathermap.org/data/2.5/forecast?q=Chennai&appid=${apiKey}&units=metric`;

    // Function to fetch forecast data
    fetch(forecastUrl)
        .then(response => response.json())
        .then(data => {
            // Filter forecast to include only one record per day
            const forecastByDay = {};
            data.list.forEach(item => {
                const date = item.dt_txt.slice(0, 10); // Extract date from forecast data
                if (!forecastByDay[date]) {
                    forecastByDay[date] = item;
                }
            });

            // Update HTML with forecast data
            const forecastContainer = document.getElementById('forecast');
            Object.values(forecastByDay).forEach(item => {
                // Create weather card for each day
                const weatherCard = document.createElement('div');
                weatherCard.classList.add('weather-card');

                // Populate weather card with data
                weatherCard.innerHTML = `
                    <div class="date">${item.dt_txt}</div>
                    <div class="weather-info">
                        <div class="description">${item.weather[0].description}</div>
                        <div class="temperature">${item.main.temp}°C</div>
                    </div>
                `;

                // Append weather card to forecast container
                forecastContainer.appendChild(weatherCard);
            });
        })
        .catch(error => console.error('Error fetching forecast:', error));
});
