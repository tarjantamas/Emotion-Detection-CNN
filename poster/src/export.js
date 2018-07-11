html2canvas(document.querySelector("#poster")).then(canvas => {
    document.body.appendChild(canvas)
});