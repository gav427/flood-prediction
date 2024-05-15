document.addEventListener('DOMContentLoaded', function () {
    const loginForm = document.getElementById('loginForm');
    loginForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        // Validation
        if (!email || !password) {
            alert('Please fill out all fields.');
            return;
        }

        // Perform login
        loginUser(email, password);
    });

    function loginUser(email, password) {
        const userData = {
            email: email,
            password: password
        };

        // Send login data to server
        fetch('http://localhost:3000/login', {
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
            if (data.isAdmin) {
                window.location.href = 'admin.html'; // Redirect to admin.html
            } else {
                window.location.href = 'client.html'; // Redirect to client.html
            }
        })        
        .catch(error => {
            console.error('There was a problem with the login:', error);
            alert('Login failed. Please try again.');
        });
    }
});
