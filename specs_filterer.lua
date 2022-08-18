---Quick utility script for picking out the top x runs based 
--on accuracy and/or time from `/specs`. 
-- @module specs_filterer
--
-- @version Lua 5.1
-- @version Lua 5.2
-- @version LuaJIT 5.2
-- 
-- @license MIT
-- @date 2022-06-27
-- 
-- @author Dulfiqar 'activexdiamond' H. Al-Safi

------------------------------ Metadata ------------------------------
local _METADATA = {
	TITLE = "C.T. Scan I.C.H. Detector",
	DESCRIPTION = "An M.L. model capable of detecting the precense of I.C.H. in 2-dimensional C.T. scans of the human brain.",
	TYPE = "CLI",
	VERSION = "dev-1.10.0",
	LICENSE = "MIT",
	AUTHOR = "Dulfiqar 'activexdiamond' H. Al-Safi"
}

local function echo_metadata()
	print( "================---~~~ INFO ~~~---================")
	print( "=> Echoing project metadata!                     =")
	print("=> Title: " .. _METADATA.TITLE .. "}             =")
	print("=> Type: " .._METADATA.TYPE .. "                                     =")
	print("=> Version: " .. _METADATA.VERSION .. "                           =")
	print("=> License: " .. _METADATA.LICENSE .. "                                  =")
	--print(f"=> Author: " .. _METADATA.AUTHOR .. "\t=")
	print( "================---~~~ ~~~~ ~~~---================")
end
echo_metadata()

------------------------------ Config ------------------------------
local INPUT_DIR = "src/specs/dev-1.12.0/"
local OUTPUT_DIR = INPUT_DIR .. "/best/"

local INPUT_FILES = {
	"svm.txt", 
	"rfc.txt",
	"dtc.txt", 
}

local ACCURACY_COUNT = 25
local TIME_COUNT = 25

------------------------------ API ------------------------------
--Iterates over `input` and filters out the (sorted) top `max` entries of `target` where `comp` is true and saves them to `output`. 
--@function [parent=#specs_filterer] extract
--@param #string input The `filename` of where to fetch values from. Should be relative to the script.
--@param #string output The `filename` of where to save the result to. Should be relative to the script.
--@param #string target The name of the field to look for. Should be in the format: `<target=valx> 
--where `x` is a single-char symbol representing the unit.
--@param #string/function comp The function used to check whether a value is better than a previously found one; 
--should return `true` if better. 
--If string; must be one of `greater` or `lesser` and the corrsponding operator will be used to compare.
--@param #number max How many entries should be stored. 
local function extract(input, output, target, comp, max)
	if type(comp) == 'string' then
		if comp == "greater" then
			comp = function(new, old) return new > old end
		elseif comp == "lesser" then
			comp = function(new, old) return new < old end
		end
	end

	local step;
	local t = {}
	for line in io.lines(input) do
		if line:sub(1, 6) == "=====>" then
			step = line:sub(16)
		end
		 
		local val = line:match(target .. "=(.-)>")
		if not val then goto continue end
		val = val:sub(1, -2)
		if #t == 0 then
			table.insert(t, {
					[target] = val,
					line = line,
					step = step,
			})
			goto continue
		end
		
		for k, v in ipairs(t) do
			comp(val, v[target])
			if comp(val, v[target]) then
				table.insert(t, k, {
					[target] = val,
					line = line,
					step = step,
				})
				if #t > max then
					table.remove(t)
				end
				goto continue
			end 
		end
		::continue::
	end
	
	local file = io.open(output, "a")
	io.write("The top " .. max .. " results gotten were as follows (in descending order):\n")
	file:write("The top " .. max .. " results gotten were as follows (in descending order):\n")
	io.write("value\t\t\t| pos\t\t| config\t\t\t| full-text\n")
	file:write("value\t\t\t| pos\t\t| config\t\t\t| full-text\n")
	local template = "%s=%f\t\t| %d\t\t| %s\t\t| %s\n"
	for k, v in ipairs(t) do
		io.write(template:format(target, v[target], k, v.step, v.line))
		file:write(template:format(target, v[target], k, v.step, v.line))
	end
	
	io.write("\n===============\n\n")
	file:write("\n===============\n\n")
	
	file:close()
end

------------------------------ Usage ------------------------------
io.stdout:setvbuf('no')

for _, f in ipairs(INPUT_FILES) do
	extract(INPUT_DIR .. f, OUTPUT_DIR .. f, "acc", "greater", ACCURACY_COUNT)
	extract(INPUT_DIR .. f, OUTPUT_DIR .. f, "total_dur", "lesser", TIME_COUNT)
end

if love then
	love.event.quit()
end






