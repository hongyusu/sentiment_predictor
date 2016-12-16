$(document).ready(function(){
    //////// init ///////////
    $('#result').hide();
    $('#partnerResult').hide();
    $('#loader').show();
    
    var airports = null;
    
    function addCities(){
	    
	var div = $('<div class="form-group dests"><label class="col-sm-2"></label><div class="col-sm-10 text-left"></div></div>').insertAfter('div.dests:last');
	for(var i=0; i<3; i++){
	    $('div.dests:last .col-sm-10').append("<input class='dest form-control pull-left box'/>");
	};
    }
    // addCities();

    $('button#moreDest').on('click', function(e){
	addCities();
	$( ".dest" ).autocomplete({
	    source: airports
	});
	e.preventDefault();
    });
    
    $("#startTime").add("#endTime").datepicker({
	dateFormat:'yy-mm-dd'
    });

    $.getJSON('/static/airports.json', function(data){
	airports = $.map(data, function(row, index){
	    return {'label': row['city'] + '(' + row['name'] + ')',
		    'value': row['city']};
	});
		    
	$( ".dest" ).autocomplete({
	    source: airports
	});
    });
    
    function getFormData(){
	var cities= $.map(
	    $('.dests input')
		.filter(function(){
		    return $(this).val().length > 0;
		}),
	    function(r){
		return $(r).val().toString();
	    });
	
	var post_data = {
	    'startTime': $('#startTime').val(),
	    'endTime': $('#endTime').val(),
	    'homeCity': 'Helsinki',
	    'cities': cities
	};
	return post_data;
    }

    var loader = $('#loader');
    $(document).ajaxStart(function(){
	loader.show();	
    });
    
    $('button#search').on('click', function(e){
	var post_data = getFormData();
	$.ajax({
	    url: '/search',
	    type: 'POST',
	    contentType:'application/json',
	    data: JSON.stringify(post_data),
	    dataType:'json',
	    success: function(data){
		//On ajax success do this		
		var w = 2000;
		var h = 50 * (data['options'].length + 1) + 50; 
		var svg_padding_left = 500;
		
		// clean stuff
		d3.select('svg').remove();
		
		var svg = d3.select('#result')
			.append('svg')
			.attr("width", w)
			.attr("height", h)
		plot_trips(svg, data['options']);
		$('#result').show();
		loader.hide();
	    },
	    error: function(xhr, ajaxOptions, thrownError) {
		if (xhr.status == 200) {
		    console.log(ajaxOptions);
		}
		else {
		    console.error(xhr.status);
		    console.error(thrownError);
		}
		loader.hide();
	    }
	});
	e.preventDefault();
    }).click();



		
		
		$("button#action1").on("click", function(e){
			var action1Data = $('#action1Input').val().toString();
			$.ajax({
				url: '/action1',
				type: 'POST',
				contentType:'application/json',
				data: JSON.stringify(action1Data),
				dataType:'json',
				success: function(d){
					$('#action1Meanscore').html(d['meanscore'].toFixed(3));
					var rows = [];
					if(d['status'] == 0){
						$.each(d['items'], function(i, item){
							rows.push([item.substring(1, 90),d['scores'][i].toFixed(3)]);
							});		    
							};
							
					// table
					d3.select('table').remove();
					var table = d3.select("#action1Plots").append("table");
				      thead = table.append("thead");
				      tbody = table.append("tbody");

				  thead.append("th").text("Score");
				  thead.append("th").text("Tweet");
				  thead.append("th").text("Curve");
				
					var tr = tbody.selectAll("tr")
					      .data(rows)
					      .enter().append("tr");

					  var td = tr.selectAll("td")
					        .data(function(d) { return [d[1],d[0]]; })
					      .enter().append("td")
					        .text(function(d) { return d; });

					  var width = 130,
					      height = d3.select("table")[0][0].clientHeight-50,
					      mx = 10,
					      radius = 2;
					
					
					
				  // Now add the chart column
				  d3.select("#action1Plots tbody tr").append("td")
				    .attr("id", "chart")
				    .attr("width", width + "px")
				    .attr("rowspan", rows.length);

				  var chart = d3.select("#chart").append("svg")
				      .attr("class", "chart")
				      .attr("width", width)
				      .attr("height", height);

				  var maxMu = 0;
  				var minMu = 1;
				/*
				  var minMu = Number.MAX_VALUE;
				  for (i=0; i < rows.length; i++) {
				    if (rows[i][1] > maxMu) { maxMu = rows[i][1]; }
				    if (rows[i][1] < minMu) { minMu = rows[i][1]; }
				  }*/
				
				  var dates = rows.map(function(t,i) { return i; });

				  var xscale = d3.scale.linear()
				    .domain([minMu, maxMu])
				    .range([mx, width-mx])
				    .nice();
				
				  var yscale = d3.scale.ordinal()
				    .domain(dates)
				    .rangeBands([0, height]);

				  chart.selectAll(".xaxislabel")
				      .data(xscale.ticks(2))
				    .enter().append("text")
				      .attr("class", "xaxislabel")
				      .attr("x", function(d) { return xscale(d); })
				      .attr("y", 5)
				      .attr("text-anchor", "middle")
				      .text(String)

				  chart.selectAll(".xaxistick")
				      .data(xscale.ticks(2))
				    .enter().append("line")
				      .attr("x1", function(d) { return xscale(d); })
				      .attr("x2", function(d) { return xscale(d); })
				      .attr("y1", 5)
				      .attr("y2", height)
				      .attr("stroke", "#eee")
				      .attr("stroke-width", 1);

				  chart.selectAll(".line")
				      .data(rows)
				    .enter().append("line")
				      .attr("x1", function(d) { return xscale(d[1]); })
				      .attr("y1", function(d,i) { return yscale(i) + yscale.rangeBand()/2; })
				      .attr("x2", function(d,i) { return rows[i+1] ? xscale(rows[i+1][1]) : xscale(d[1]); })
				      .attr("y2", function(d,i) { return rows[i+1] ? yscale(i+1) + yscale.rangeBand()/2 : yscale(i) + yscale.rangeBand()/2; })
				      .attr("stroke", "#777")
				      .attr("stroke-width", 1);

				  var pt = chart.selectAll(".pt")
				      .data(rows)
				    .enter().append("g")
				      .attr("class", "pt")
				      .attr("transform", function(d,i) { return "translate(" + xscale(d[1]) + "," + (yscale(i) + yscale.rangeBand()/2) + ")"; });

				  pt.append("circle")
				      .attr("cx", 0)
				      .attr("cy", 0)
				      .attr("r", radius)
				      .attr("opacity", .5)
				      .attr("fill", "#ff0000");
								
								
				$('#action1Plot').show();
				
				
				
	    },
	    error: function(xhr, ajaxOptions, thrownError) {
		if (xhr.status == 200) {
		    console.log(ajaxOptions);
		}
		else {
		    console.error(xhr.status);
		    console.error(thrownError);
		}
		loader.hide();
	    }
	});
	e.preventDefault();
    }).click();
	
});

