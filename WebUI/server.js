// this is the new code 
const http = require('http');
const fs = require('fs');
const path = require('path');
const nodemailer = require('nodemailer');

const server = http.createServer((req, res) => {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        // Respond to preflight requests
        res.writeHead(200);
        res.end();
        return;
    }

    if (req.method === 'POST') {
        if (req.url === '/register') {
            handleRegistration(req, res);
        } else if (req.url === '/login') {
            handleLogin(req, res);
        } else if (req.url === '/send-email') {
            handleEmail(req, res); // Handle email sending
        } else {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('Page not found');
        }
    } else {
        res.writeHead(405, { 'Content-Type': 'text/plain' });
        res.end('Method Not Allowed');
    }
});

function handleRegistration(req, res) {
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', () => {
        const userData = JSON.parse(body);
        userData.isAdmin = false;
        saveUserData(userData, (err) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Internal Server Error' }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ message: 'Registration successful' }));
            }
        });
    });
}

function handleLogin(req, res) {
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', () => {
        const loginData = JSON.parse(body);
        authenticateUser(loginData, (err, user) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Internal Server Error' }));
            } else {
                if (user) {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ message: 'Login successful', isAdmin: user.isAdmin }));
                } else {
                    res.writeHead(401, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Unauthorized' }));
                }
            }
        });
    });
}

function handleEmail(req, res) {
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', () => {
        const { content } = JSON.parse(body);

        // Read users data from users.json
        const usersFilePath = path.join(__dirname, 'users.json');
        const usersData = JSON.parse(fs.readFileSync(usersFilePath));

        // Create Ethereal transporter
        const transporter = nodemailer.createTransport({
            host: 'smtp.ethereal.email',
            port: 587,
            auth: {
                user: 'oren.jacobson@ethereal.email',
                pass: 'dFqPEw4KJhgAUrFzBG'
            }
        });

        // Loop through users and send email to each
        usersData.forEach(user => {
            const mailOptions = {
                from: 'oren.jacobson@ethereal.email',
                to: user.email,
                subject: 'Flood Alert',
                text: content
            };

            transporter.sendMail(mailOptions, (error, info) => {
                if (error) {
                    console.error('Error sending email to', user.email, ':', error);
                } else {
                    console.log('Email sent to', user.email, ':', info.response);
                }
            });
        });

        // Send a response to the client
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ message: 'Emails sent successfully' }));
    });
}

function saveUserData(userData, callback) {
    const usersFilePath = path.join(__dirname, 'users.json');
    fs.readFile(usersFilePath, (err, data) => {
        if (err && err.code !== 'ENOENT') {
            callback(err);
            return;
        }

        const users = JSON.parse(data || '[]');
        users.push(userData);

        fs.writeFile(usersFilePath, JSON.stringify(users, null, 2), callback);
    });
}

function authenticateUser(loginData, callback) {
    const usersFilePath = path.join(__dirname, 'users.json');
    fs.readFile(usersFilePath, (err, data) => {
        if (err) {
            callback(err);
            return;
        }

        const users = JSON.parse(data || '[]');
        const user = users.find(u => u.email === loginData.email && u.password === loginData.password);

        if (user) {
            callback(null, { ...user, isAdmin: user.isAdmin }); // Include isAdmin field in the response
        } else {
            callback(null, null);
        }
    });
}


const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
