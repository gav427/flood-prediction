document.addEventListener('DOMContentLoaded', function () {
    const registrationForm = document.getElementById('registrationForm');
    registrationForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const fullName = document.getElementById('fullName').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const region = document.getElementById('region').value;

        // Validation
        if (!fullName || !email || !password || !confirmPassword || !region) {
            alert('Please fill out all fields.');
            return;
        }

        if (password !== confirmPassword) {
            alert('Passwords do not match.');
            return;
        }

        // Proceed with registration
        registerUser(fullName, email, password, region);
    });

    function registerUser(fullName, email, password, region) {
        const userData = {
            fullName: fullName,
            email: email,
            password: password,
            region: region
        };

        // Send data to server
        fetch('http://localhost:3000/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok.');
        })
        .then(data => {
            console.log(data.message); // Log success message from server
            alert('Registration successful!');
            window.location.href = 'login.html';
        })
        .catch(error => {
            console.error('There was a problem with the registration:', error);
            alert('Registration failed. Please try again.');
        });
    }
});
