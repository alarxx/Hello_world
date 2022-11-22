const fortunes = ["one","two","three","four","five", "six"];
module.exports = () => fortunes[Math.floor(Math.random() * fortunes.length)];
module.exports.obj = "looll";
