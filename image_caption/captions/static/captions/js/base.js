document.getElementById('captionForm').onsubmit = function(e) {
    e.preventDefault();

    const form = new FormData(this);
    const fileInput = document.getElementById('media');
    const file = fileInput.files[0];
    if (!file) return;

    const type = file.type.startsWith('video') ? 'video' : 'image';
    form.delete('image');
    form.append(type, file);

    // Show loading, hide button
    document.getElementById('generateBtn').style.display = 'none';
    document.getElementById('loadingMessage').style.display = 'inline';

    fetch("/generate-caption/", {
        method: "POST",
        body: form,
        headers: {
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('generateBtn').style.display = 'inline';
        document.getElementById('loadingMessage').style.display = 'none';

        if (data.status === 'success') {
            document.getElementById('captionText').innerText = data.caption;
            const preview = document.getElementById('mediaPreview');
            preview.innerHTML = type === 'image'
                ? `<img src="${data.media_url}" class="img-fluid" style="max-width: 400px;">`
                : `<video src="${data.media_url}" controls style="max-width: 400px;"></video>`;
            document.getElementById('result').style.display = 'block';
        } else {
            alert(data.message);
        }
    })
    .catch(err => {
        document.getElementById('generateBtn').style.display = 'inline';
        document.getElementById('loadingMessage').style.display = 'none';
        alert('Something went wrong! ' + err);
    });
};

// CSRF helper
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
