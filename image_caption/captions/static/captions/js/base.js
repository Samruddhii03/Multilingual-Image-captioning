
document.getElementById('captionForm').onsubmit = function(e) {
    e.preventDefault();
    const formData = new FormData(this);

    fetch("{% url 'generate_caption_view' %}", {
        method: "POST",
        body: formData,
        headers: {
            'X-CSRFToken': '{{ csrf_token }}'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('captionText').innerText = data.caption;
            document.getElementById('uploadedImage').src = data.image_url;
            document.getElementById('result').style.display = 'block';
        } else {
            alert("Error: " + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
};
