function plot_trips(svg, data){
    var IMAGE = true;
    var all_dts = [];
    var padding_left = 100;
    var W = parseInt(svg.attr('width')) - padding_left;
    
    var H = parseInt(svg.attr('height'));
        
    $.each(data, function(row_i, row){
	row['id'] = row_i;
	$.each(row['trips'], function(i, trip){
	    console.assert(trip['startTime'] != undefined, trip);
	    console.assert(trip['endTime'] != undefined, trip);
	    trip['startTime'] = parse_datetime(trip['startTime']);
	    trip['endTime'] = parse_datetime(trip['endTime']);
	    trip['row'] = row_i;
	    all_dts.push(trip['startTime']);
	    all_dts.push(trip['endTime']);
	});
    });
    var min_dt = all_dts.reduce(function (a, b) { return a < b ? a : b; });
    var max_dt = all_dts.reduce(function (a, b) { return a > b ? a : b; });
    // console.log('max_dt, min_dt:', min_dt, max_dt);
    $.each(data, function(i, row){
	$.each(row['trips'], function(i, trip){
	    // console.log(min_dt, max_dt, trip['startTime'], trip['endTime'],
	    // 		scaled_position_of_time(min_dt, max_dt, trip['startTime'], W),
	    // 	       scaled_position_of_time(min_dt, max_dt, trip['endTime'], W));
	    trip['x1'] = scaled_position_of_time(min_dt, max_dt, trip['startTime'], W);
	    trip['x2'] = scaled_position_of_time(min_dt, max_dt, trip['endTime'], W);
	});
    });

    // add trip events
    var rec_h = 50;
    var paddingLeft = 20;
    var firstPaddingTop = 40;
    var paddingTop = 10;
    var padding_left_attr = 'translate(' + padding_left + ', 0)';
    var color_map = {
	'flight': '#3182bd',
	'hotel': '#C0E1F9',
	'site': '#FF8300',
    }
    var html_map = {
	'flight': function(d){
	    return "Flying from " + "<strong>" + d['startCity'] + "</strong>" + " to " + "<strong>" + d['endCity'] + "</strong>";
	},
	'hotel': function(d){
	    return "At " + "<strong>" + d['name'] + "</strong>";
	},
	'site': function(d){
	    return "At " + "<strong>" + d['name'] + "</strong>";
	}
    };
    var tip = d3.tip()
	    .attr('class', 'd3-tip')
	    .offset([-5, 0])
	    .html(function(d) {
		return html_map[d['type']](d);
	    });
    svg.call(tip);

    // background covers
    svg
	.selectAll('rect')
	.data(data)
	.enter()
	.append('rect')
	.classed('row_cover', true)
	.attr('height', rec_h)
	.attr('width', W)
	.attr('y', function(d, i){
	    var padding_top = 0;
	    if(i==0){
		return firstPaddingTop + padding_top;
	    }
	    else{
		return firstPaddingTop + (paddingTop + rec_h) * i + padding_top;
	    }  
	})
	.attr('fill', function(d, i){
	    if(i % 2){		
		return '#E5F4FC';
	    }
	    else{
		return '#ffffff';
	    }
	})
	.attr('opacity', 0.5)
	.on('mouseover', function(d, i){
	    d3.select(this).classed('active', true);
	})
	.on('mouseout', function(d, i){
	    d3.select(this).classed('active', false);
	})
	.on('click', function(d, i){
	    $('#tripDetails').modal();
	});
    

    svg
	.selectAll('g')
	.data(data)
	.enter()
	.append('g')
	.classed('events', true)
	.attr('transform', padding_left_attr)
	.selectAll('rect')
	.data(function(d, i){
	    return d['trips'];
	})
	.enter()
	.append('rect')
	.attr('fill', function(t, i){
	    return color_map[t['type']];
	})
	.attr('opacity', 0.8)
	.attr('height', rec_h)
	.attr('width', function(t){
	    return t['x2'] - t['x1'];
	})
	.attr('x', function(t){
	    return t['x1'];
	})
    	.attr('y', function(t, i){
	    if(t['row']==0){
		return firstPaddingTop;
	    }
	    else{
		return firstPaddingTop + (paddingTop + rec_h) * t['row'];
	    }
	})
    	.on('mouseover', tip.show)
	.on('mouseout', tip.hide);

    svg
	.append('g')
	.classed('cost_column', true)
	.selectAll('text')
	.data(data)
	.enter()
	.append('text')
	.classed('cost_row', true)
	.attr('x', function(d){
	    return 14;
	})
    	.attr('y', function(d, i){
	    var padding_top = 28;
	    if(i==0){
		return firstPaddingTop + padding_top;
	    }
	    else{
		return firstPaddingTop + (paddingTop + rec_h) * i + padding_top;
	    }
	})
	.text(function(d){
	    return parseInt(d['totalPrice']) + '€';
	})
    .attr('fill', function(d, i){
	if(i==0){
	    return '#449D44';
	}
	else{
	    return '#aaa';
	}
    });


    if(IMAGE){
    
	var icon_map = {
	    'flight': "/static/images/flight.svg",
	    'hotel': "/static/images/hotel.svg",
	    // 'site': "/static/images/site.svg",
	}

	svg.selectAll('g.events')
	    .selectAll('image')
	    .data(function(d, i){
		return d['trips'];
	    })
	    .enter()
	    .append('image')
	    .attr('x', function(t){
		var offset = (t['x2'] - t['x1']) / 2 - 12;
		return t['x1'] + offset;
	    })
    	    .attr('y', function(t, i){
		var offset = rec_h / 2 - 12;
		if(t['row']==0){
		    return firstPaddingTop + offset;
		}
		else{
		    return firstPaddingTop + (paddingTop + rec_h) * t['row'] + offset;
		}
	    })
	    .attr('height', 24)
	    .attr('width', 24)
	    .attr('xlink:href', function(t){
		return icon_map[t['type']];
	    })
	    .attr('opacity', 0.5)
            .on('mouseover', tip.show)
	    .on('mouseout', tip.hide);
    }

    // add anchor date text
    var anchor_dts = time_anchors_between_period(min_dt, max_dt);
    var anchor_pts = $.map(anchor_dts, function(dt){
	return {
	    'dt': dt.getMonth() + '-' + dt.getDate() + ' ' + dt.getHours() + 'am',
	    'x': scaled_position_of_time(min_dt, max_dt, dt, W)
	};
    });
    
    svg.append('g')
	.selectAll('text')
	.data(anchor_pts)
	.enter()
	.append('text')
    	.classed('anchor-time', true)
	.attr('transform', padding_left_attr)
	.attr('y', 30)
	.attr('x', function(pt){
	    return pt['x']-5;
	})
	.text(function(pt){
	    return pt['dt'];
	});
    // add anchor date vertical line
    svg.append('g')
	.selectAll('line')
	.data(anchor_pts)
	.enter()
	.append('line')
	.attr('transform', padding_left_attr)
	.attr('x1', function(pt){
	    return pt['x'];
	})
	.attr('x2', function(pt){
	    return pt['x'];
	})
	.attr('y1', 33)
	.attr('y2', H)
	.attr('stroke', '#eee')
	.attr('stroke-width', 1);
}

$(document).ready(function(){
    // $.getJSON("/static/example_trips.json", function(data){
    // 	var w = 1000;
    // 	var h = 50 * (data['options'].length + 1); 
    // 	var svg = d3.select('#content')
    // 		.append('svg')
    // 		.attr("width", w)
    // 		.attr("height", h);

    // 	plot_trips(svg, data['options']);
    // });
});

function test(svg){
    svg.selectAll('rect')
	.data([1, 2, 3])
	.enter()
	.append('rect')
	.attr('width', 50)
	.attr('height', 10)
	.attr('x', function(x){
	    console.log(x);
	    return x * 100;
	})
	.attr('fill', '#666');

}