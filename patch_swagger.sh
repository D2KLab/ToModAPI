#!/usr/bin/env bash
b=$(pip show swagger-ui-py | grep Location)
loc="$(cut -d':' -f2 <<<$b)"
loc=$(echo $loc | tr -d ' ')
loc+=/swagger_ui/core.py
sed -i $loc -e "s/@swagger_blueprint.route(r'')/@swagger_blueprint.route(r'\/')/g" $loc
