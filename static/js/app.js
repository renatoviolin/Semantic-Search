var data = []
var token = ""

function truncateString(str) {
    num = 85
    if (str.length <= num)
        return str
    return str.slice(0, num) + '...'
}
function truncateScore(str) {
    num = 6
    return String(str).slice(0, num)
}

jQuery(document).ready(function () {
    var slider = $('#max_sentences')
    slider.on('change mousemove', function (evt) {
        $('#label_max_sentences').text('Top k sentences: ' + slider.val())
    })

    $('#btn-process').on('click', function () {

        $.ajax({
            url: '/get_predictions',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_corpus": $('#input_corpus').val(),
                "input_query": $('#input_query').val(),
                "split_token": $('#split_token').val(),
                "top_k": slider.val(),
            }),
            beforeSend: function () {
                $('#output').hide()
                $('.overlay').show()
                $('#output-use').empty()
                $('#output-bm25').empty()
                $('#output-sentenceBERT').empty()
                $('#output-bert').empty()
                $('#output-roberta').empty()
                $('#output-infersent').empty()
            },
            complete: function () {
                $('.overlay').hide()
                $('#output').show()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            for (i = 0; i < jsondata['use_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['use_scores'][i])}</b><br>
                    ${truncateString(jsondata['use_sentences'][i])}</p>`
                $('#output-use').append(el)
            }

            for (i = 0; i < jsondata['bm25_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['bm25_scores'][i])}</b><br>    
                    ${truncateString(jsondata['bm25_sentences'][i])}</p>`
                $('#output-bm25').append(el)
            }

            for (i = 0; i < jsondata['sentenceBERT_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['sentenceBERT_scores'][i])}</b><br>
                    ${truncateString(jsondata['sentenceBERT_sentences'][i])}</p>`
                $('#output-sentenceBERT').append(el)
            }

            for (i = 0; i < jsondata['infersent_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['infersent_scores'][i])}</b><br>
                    ${truncateString(jsondata['infersent_sentences'][i])}</p>`
                $('#output-infersent').append(el)
            }

            for (i = 0; i < jsondata['bert_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['bert_scores'][i])}</b><br>
                    ${truncateString(jsondata['bert_sentences'][i])}</p>`
                $('#output-bert').append(el)
            }

            for (i = 0; i < jsondata['roberta_sentences'].length; i++) {
                cor = i % 2
                el = `<p class='cor_${cor}'>
                    <b>Score: ${truncateScore(jsondata['roberta_scores'][i])}</b><br>
                    ${truncateString(jsondata['roberta_sentences'][i])}</p>`
                $('#output-roberta').append(el)
            }

        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })
})