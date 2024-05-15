document.addEventListener('DOMContentLoaded', function () {
    // API key
    const apiKey = '46307c0733115117bdbc3aa38ad0028a';

    // City for forecast
    const city = 'Chennai';

    // URL for forecast
    const forecastUrl = `https://api.openweathermap.org/data/2.5/forecast?q=${city}&appid=${apiKey}&units=metric`;

    // Function to fetch forecast data
    fetch(forecastUrl)
        .then(response => response.json())
        .then(data => {
            // Filter forecast to include only rainfall predictions for the next 7 days
            const rainfallPredictions = [];
            data.list.forEach(item => {
                const date = new Date(item.dt_txt).toLocaleDateString(); // Extract date from forecast data
                const rainfall = item.rain ? item.rain['3h'] : 0; // Check if rainfall data is available
                rainfallPredictions.push({ date, rainfall });
            });

            // Get the next 7 days
            const next7Days = Array.from({ length: 7 }, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() + i + 1);
                return date.toLocaleDateString();
            });

            // Prepare data for the chart
            const labels = next7Days;
            const rainfallData = next7Days.map(day => {
                const prediction = rainfallPredictions.find(prediction => prediction.date === day);
                return prediction ? prediction.rainfall : 0;
            });

            // Create the chart
            const ctx = document.getElementById('rainfallChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Rainfall Prediction (mm)',
                        data: rainfallData,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        yAxes: [{
                            ticks: {
                                beginAtZero: true
                            }
                        }]
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching forecast:', error));
});
// function sendEmail() {
//     const emailContent = document.getElementById('email-content').value;

//     // Create an object with the email content
//     const emailData = { content: emailContent };

//     // Make an HTTP POST request to your server endpoint
//     fetch('http://localhost:3000/send-email', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(emailData)
//     })
//     .then(response => {
//         if (response.ok) {
//             alert('Email sent successfully!');
//         } else {
//             throw new Error('Failed to send email');
//         }
//     })
//     .catch(error => {
//         console.error('Error sending email:', error);
//         alert('Failed to send email');
//     });
// }
async function sendEmail() {
    const emailContent = document.getElementById('email-content').value;

    // Create an object with the email content
    const emailData = { content: emailContent };

    try {
        // Make an asynchronous HTTP POST request to your server endpoint
        const response = await fetch('http://localhost:3000/send-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(emailData)
        });

        if (response.ok) {
            alert('Email sent successfully!');
        } else {
            throw new Error('Failed to send email');
        }
    } catch (error) {
        console.error('Error sending email:', error);
        alert('Failed to send email');
    }
}


