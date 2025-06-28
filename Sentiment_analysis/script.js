
// Function to handle option selection and display input fields accordingly
document.addEventListener('DOMContentLoaded', function () {
    console.log("DEBUG: DOM Content Loaded"); // Verify that the DOM is loaded

    const inputSection = document.getElementById('input-section');
    const outputSection = document.getElementById('output-section');
    const loadingIndicator = document.getElementById('loading-indicator'); // Ensure this is in your HTML

    outputSection.style.display = 'none'; // Hide output initially

    // Function to handle the selection of the analysis option
    function selectOption(option) {
        inputSection.innerHTML = ''; // Clear previous inputs

        // Highlight the selected option
        document.querySelectorAll('.option').forEach(el => el.classList.remove('selected'));
        const selectedElement = document.querySelector(`.option[onclick="selectOption('${option}')"]`);
        if (selectedElement) {
            selectedElement.classList.add('selected');
        }

        if (option === 'twitter') {
            inputSection.innerHTML = `
                <h3>Twitter Analysis</h3>
                <input type="number" id="tweet-limit" placeholder="Number of Tweets" required>
                <input type="text" id="tweet-domain" placeholder="Keyword for Tweets" required>
                <select id="language-select">
                    <option value="">Select Language (optional)</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <!-- Add more language options as needed -->
                </select>
            `;
        } else if (option === 'custom_text') {
            inputSection.innerHTML = `
                <h3>Custom Text Analysis</h3>
                <textarea id="custom-text" placeholder="Enter your custom text" required></textarea>
            `;
        } else if (option === 'instagram') {
            inputSection.innerHTML = `<h3>Instagram Analysis (Coming Soon)</h3>`;
        }
    }

    // Function to handle the submission of data to the backend
    async function analyzeSentiment() {
        const selectedOptionElement = document.querySelector('.option.selected');
        if (!selectedOptionElement) {
            alert('Please select an analysis option first.');
            return;
        }

        const selectedOption = selectedOptionElement.getAttribute('onclick').split("'")[1];
        let url = '/analyze'; // Your Flask API endpoint
        let data = {};

        if (selectedOption === 'twitter') {
            const tweetLimit = document.getElementById('tweet-limit').value;
            const keyword = document.getElementById('tweet-domain').value;
            const language = document.getElementById('language-select').value;

            if (!tweetLimit || !keyword) {
                alert('Please fill in all required fields for Twitter analysis.');
                return;
            }

            data = {
                data_source: 'twitter',
                tweet_limit: tweetLimit,
                keyword: keyword,
                language: language || null // Optional
            };
        } else if (selectedOption === 'custom_text') {
            const customText = document.getElementById('custom-text').value;

            if (!customText) {
                alert('Please fill in the text for Custom Text analysis.');
                return;
            }

            data = {
                data_source: 'custom_text',
                custom_text: customText
            };
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';

        // Send POST request to the backend
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Failed to get a valid response from the server.');
            }

            const result = await response.json();
            console.log("DEBUG: Backend Response", result); // Log the response for debugging

            // Check if data is from Twitter and call the respective function
            if (data.data_source === 'twitter') {
                displayTwitterResults(result);
            } else {
                displayResults(result);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing sentiment. Please try again.');
        } finally {
            // Hide loading indicator after processing
            loadingIndicator.style.display = 'none';
        }
    }

    // Function to display results for Twitter analysis
    function displayTwitterResults(data) {
        outputSection.style.display = 'block'; // Show the output section

        // Clear previous results
        outputSection.innerHTML = '<h3>Twitter Analysis Results</h3>';

        // Display the analysis summary
        if (data.message) {
            const message = document.createElement('p');
            message.textContent = data.message;
            outputSection.appendChild(message);
        }

        // Display multiple sentiment distribution charts if available
        if (data.distribution && Array.isArray(data.distribution)) {
            renderMultipleSentimentCharts(data.distribution);
        } else if (data.distribution) {
            // If only a single distribution is available
            renderSentimentChart(data.distribution);
        }

        // Display individual sentiment results if available
        if (data.sentiments && data.sentiments.length > 0) {
            const sentimentsList = document.createElement('ul');
            data.sentiments.forEach(sentiment => {
                const listItem = document.createElement('li');
                listItem.textContent = `${sentiment.label}: ${sentiment.tweet}`;
                listItem.classList.add(sentiment.label.toLowerCase());
                sentimentsList.appendChild(listItem);
            });
            outputSection.appendChild(sentimentsList);
        } else {
            const noResults = document.createElement('p');
            noResults.textContent = 'No detailed sentiment analysis results available.';
            outputSection.appendChild(noResults);
        }
    }

    // Function to display the results for general analysis
    function displayResults(data) {
        outputSection.style.display = 'block'; // Show the output section

        // Clear previous results
        outputSection.innerHTML = '<h3>Analysis Results</h3>';

        // Display the analysis summary
        if (data.message) {
            const message = document.createElement('p');
            message.textContent = data.message;
            outputSection.appendChild(message);
        }

        // Display the sentiment distribution chart if available
        if (data.distribution) {
            renderSentimentChart(data.distribution);
        }
    }

    // Function to render multiple sentiment charts using Chart.js
    function renderMultipleSentimentCharts(distributions) {
        distributions.forEach((distribution, index) => {
            const ctx = document.createElement('canvas');
            outputSection.appendChild(ctx);

            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        data: [distribution.positive, distribution.negative, distribution.neutral],
                        backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Sentiment Distribution ${index + 1}`
                        }
                    }
                }
            });
        });
    }

    // Function to render a single sentiment chart using Chart.js
    function renderSentimentChart(distribution) {
        const ctx = document.createElement('canvas');
        outputSection.appendChild(ctx);

        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [distribution.positive, distribution.negative, distribution.neutral],
                    backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Sentiment Distribution'
                    }
                }
            }
        });
    }

    // Attach the functions to the global scope for use in the HTML
    window.selectOption = selectOption;
    window.analyzeSentiment = analyzeSentiment;
});
