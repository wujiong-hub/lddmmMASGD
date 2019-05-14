#!/usr/bin/perl

#************************INPUTS TO THIS SCRIPT****************************
#
#
#	1) template T1 image	(moving image) 
#	2) target T1 image	(fixed image)
#	3) output folder
#	4) do a global histogram matching before AIR matrix calculations(1-yes, 0-no)
#	5) output image name 	(optional)
#
#


$BIN_DIRECTORY = $ARGV[0];
#executables used
$IMG_histmatch4 		= "$BIN_DIRECTORY\/IMG_histmatch4";
$IMG_normalization2int16 	= "$BIN_DIRECTORY\/IMG_normalization2int16";
$IMG_mask 			= "$BIN_DIRECTORY\/IMG_mask";
$IMG_saveimgsize 		= "$BIN_DIRECTORY\/IMG_saveimgsize";
$IMG_apply_AIR_tform		= "$BIN_DIRECTORY\/IMG_apply_AIR_tform";
$AIR_ALIGNLINEAR 		= "$BIN_DIRECTORY\/alignlinear";
$AIR_RESLICE     		= "$BIN_DIRECTORY\/reslice";
$AIR_COMBINEAIR     		= "$BIN_DIRECTORY\/combine_air";
$AIR_SCANAIR     		= "$BIN_DIRECTORY\/scanair";




	$no_of_inputs = $#ARGV + 1;

#Printing input arguments
        print "\nNumber of input arguments = \n";
	print "\t$no_of_inputs\n";
        print "Parameters = \n";
        for($i=0; $i<=$#ARGV; $i++) {
                print "\t$i = $ARGV[$i]\n";
        }
        print "\n";

if (($no_of_inputs < 5) || ($no_of_inputs > 6)){
	print "Wrong number of inputs\n";
	print "EXITING...\n";
	exit;
}
#READING INPUTS
	$template_img		= $ARGV[1];
	$target_img		    = $ARGV[2];	
	$output_path	  	= $ARGV[3];	
	$do_global_hist   	= $ARGV[4];	
	if ($no_of_inputs == 6){
		$output_img	    = $ARGV[5];				
	} 
	
#removing paths from filenames
	$template_path	= $template_img;
	chomp $template_img;
	$template_img	=~ /^.+\/(.+)/;
	$template_img	= $1;
	$template	= $template_img;
	$template	=~ s/.img//;
	$template_path	=~ s/\/$template_img//;

        $target_path  = $target_img;
        chomp $target_img;
        $target_img   =~ /^.+\/(.+)/;
        $target_img   = $1;
        $target       = $target_img;
        $target       =~ s/.img//;
        $target_path  =~ s/\/$target_img//;

        if ($no_of_inputs == 5){
                $output = "$template\_to\_$target";
        }
	if ($no_of_inputs == 6){
		$output = $output_img;
		$output =~ s/.img//;
	}


print "*************************************************\n";
print "*********************STEP1***********************\n";
print "*************************************************\n\n";
#Printing input arguments and checking inputs and outputs

	$FILE_ERROR = 0;

	print "0) Template   = \n";
	$temp1 = "$template_path/$template\.img";
	if (-e "$temp1"){	print "\ttemplate img = $temp1 \t\t FILE EXISTS\n";				}
	else		{	print "\ttemplate img = $temp1 \t\t FILE DOESN'T EXIST\n";	$FILE_ERROR =1;	}
	$temp1 = "$template_path/$template\.hdr";
	if (-e "$temp1"){	print "\ttemplate hdr = $temp1 \t\t FILE EXISTS\n";				}
	else		{	print "\ttemplate hdr = $temp1 \t\t FILE DOESN'T EXIST\n";	$FILE_ERROR =1;	}
	
	print "1) Target   = \n";
	$temp1 = "$target_path/$target\.img";
	if (-e "$temp1"){	print "\ttarget img   = $temp1 \t\t FILE EXISTS\n";				}
	else		{	print "\ttarget img   = $temp1 \t\t FILE DOESN'T EXIST\n";	$FILE_ERROR =1;	}
	$temp1 = "$target_path/$target\.hdr";
	if (-e "$temp1"){	print "\ttarget hdr   = $temp1 \t\t FILE EXISTS\n";				}
	else		{	print "\ttarget hdr   = $temp1 \t\t FILE DOESN'T EXIST\n";	$FILE_ERROR =1;	}

	$DIRECTORY_ERROR = 0;
	print "2) Output folder   = \n";
	mkdir("$output_path");
	if (-d "$output_path"){
		print "\toutput path  = $output_path\t\t DIRECTORY EXISTS\n";			
	}
	else		      {
		print "\toutput path  = $output_path\t\t DIRECTORY DOESN'T EXIST AND CANNOT BE CREATED \n";
		$DIRECTORY_ERROR = 1;
	}

	print "3) Do histogram matching before AIR   = \n";
	if ($do_global_hist==1)	{	print "\tYES\n";	}
	else			{       print "\tNO\n";		}

        print "4) Output   = \n";
        print "\toutput img = $output\.img \n\n";

	if($FILE_ERROR == 1){
	        print "Error in input filename or directory \n";
		print "EXITING...\n";
        	exit;		
	}	
	if($DIRECTORY_ERROR == 1){
	        print "Output directory doesn't exist and can't be created\n";
		print "EXITING...\n";
        	exit;		
	}	



print "*************************************************\n";
print "*********************STEP2***********************\n";
print "*************************************************\n\n";
print "CREATING OUTPUT TEMPORARY DIRECTORY AND COPYING FILES\n\n";

#creating output subdirectories
	$subdirectory1 = "$template\_to\_$target.tmp";
	$full_subdirectory1 = "$output_path\/$subdirectory1";
	$sim1 = `mkdir "$full_subdirectory1"`;

#copying img files and header files
        $temp1 = "$template_path/$template\.img";	
        $sim1 = `cp $temp1 "$full_subdirectory1/"`;
        $temp1 = "$template_path/$template\.hdr";	
        $sim1 = `cp $temp1 "$full_subdirectory1/"`;

        $temp1 = "$target_path/$target\.img";	
        $sim1 = `cp $temp1 "$full_subdirectory1/"`;
        $temp1 = "$target_path/$target\.hdr";	
        $sim1 = `cp $temp1 "$full_subdirectory1/"`;

#changing current directory to $output_folder
	chdir $output_path if $output_path;


print "*************************************************\n";
print "*********************STEP3***********************\n";
print "*************************************************\n\n";
#global histogram matching

$template_img1 = "$subdirectory1/$template\.img";
$target_img1   = "$subdirectory1/$target\.img";
$template_img2 = "$subdirectory1/$template\_h.img";
$target_img2   = "$subdirectory1/$target\_h.img";


if ( $do_global_hist == 1 ) {
print "GLOBAL HISTOGRAM MATCHING\n\n";

	$template_hist1 = "$subdirectory1/$template\.hist";
	$target_hist1   = "$subdirectory1/$target\.hist";
	$template_hist2 = "$subdirectory1/$template\_h.hist";
	$target_hist2   = "$subdirectory1/$target\_h.hist";

#setting thread number
        $ENV{OMP_NUM_THREADS} = '4';	
#	$sim1 = `$IMG_histmatch4 $template_img1 $target_img1 $template_img2 $target_img2 256 0 0 1 $template_hist1 $target_hist1 $template_hist2 $target_hist2`; 
	$sim1 = `$IMG_histmatch4 $template_img1 $target_img1 $template_img2 $target_img2 1024 3 0 1 $template_hist1 $target_hist1 $template_hist2 $target_hist2`; 
	print "$sim1\n";
}
else {
	$sim1 = `cp $template_img1 $template_img2`;
	$sim1 = `cp $target_img1 $target_img2`;
	$template_img1 = "$subdirectory1/$template\.hdr";
	$target_img1   = "$subdirectory1/$target\.hdr";
	$template_img2 = "$subdirectory1/$template\_h.hdr";
	$target_img2   = "$subdirectory1/$target\_h.hdr";
	$sim1 = `cp $template_img1 $template_img2`;
	$sim1 = `cp $target_img1 $target_img2`;
}



print "*************************************************\n";
print "*********************STEP4***********************\n";
print "*************************************************\n\n";
print "AIR CALCULATIONS\n\n";

#rigid affine transformation


	$template_img1 = "$subdirectory1/$template\_h.img";
	$target_img1   = "$subdirectory1/$target\_h.img";
	$template_img2 = "$subdirectory1/$template\_hi.img";
	$target_img2   = "$subdirectory1/$target\_hi.img";
	$template_img3_mask  = "$subdirectory1/$template\_hia_mask.img";
	$template_img3_noisy  = "$subdirectory1/$template\_hia_noisy.img";
	$template_img3  = "$subdirectory1/$template\_hia.img";

	$sim1 = `$IMG_normalization2int16 $template_img1 $template_img2`;
#print "$sim1\n";
	$sim1 = `$IMG_normalization2int16 $target_img1 $target_img2`;
#print "$sim1\n";

	$sim1 = `$AIR_ALIGNLINEAR $target_img2 $template_img2 "$subdirectory1/temp1.air" -m 6 -x 3 -b1 8 8 8 -b2 8 8 8 -t1 5 -t2 5`;
#print "$sim1\n";
	$sim1 = `$AIR_RESLICE "$subdirectory1/temp1.air" "$subdirectory1/temp1.img" -a $template_img2 -o -k -n 1`;
	$sim1 = `$AIR_ALIGNLINEAR $target_img2 "$subdirectory1/temp1.img" "$subdirectory1/temp2.air"  -m 6 -x 3 -b1 2 2 2 -b2 2 2 2 -t1 5 -t2 5`;
	$sim1 = `$AIR_RESLICE "$subdirectory1/temp2.air" "$subdirectory1/temp2.img" -a "$subdirectory1/temp1.img" -o -k -n 1`;
	$sim1 = `$AIR_ALIGNLINEAR $target_img2 "$subdirectory1/temp2.img" "$subdirectory1/temp3.air" -m 12 -x 3 -b1 2 2 2 -b2 2 2 2 -t1 5 -t2 5`;
	$sim1 = `$AIR_RESLICE "$subdirectory1/temp3.air" "$subdirectory1/temp3.img" -a "$subdirectory1/temp2.img" -o -k -n 1`;
	$sim1 = `$AIR_COMBINEAIR "$subdirectory1/final.air" y "$subdirectory1/temp3.air" "$subdirectory1/temp2.air" "$subdirectory1/temp1.air"`;

#FOR TESTING
#	$sim1 = `$AIR_RESLICE "$subdirectory1/final.air" $template_img3_mask -a $template_img2 -o -k -n 1`;
#	$sim1 = `$AIR_RESLICE "$subdirectory1/final.air" $template_img3_noisy -a $template_img2 -o -k -n 10`;
#	$sim1 = `$IMG_mask $template_img3_noisy $template_img3_mask $template_img3`;

	$template_img1 = "$subdirectory1/$template\.img";
	$template_imgsize1 = "$subdirectory1/$template\.imgsize";
	$template_img2 = "$subdirectory1/$template\_d.img";

	$sim1 = `$AIR_SCANAIR "$subdirectory1/final.air" > "$subdirectory1/final_air.txt"`;
	$sim1 = `$IMG_saveimgsize $template_img1 $template_imgsize1`;
	$sim1 = `$IMG_apply_AIR_tform $template_img1 $template_img2 "$subdirectory1/final_air.txt" 1 $template_imgsize1 1`;	
print "$sim1\n";


print "*************************************************\n";
print "*********************STEP5***********************\n";
print "*************************************************\n\n";
print "CLEANING TEMPORARY FILES\n\n";

	$template_img1 = "$subdirectory1/$template\_d.img";
	$template_hdr1 = "$subdirectory1/$template\_d.hdr";
	$sim1 = `cp $template_img1 "$output\.img"`;
	$sim1 = `cp $template_hdr1 "$output\.hdr"`;
	$sim1 = `cp "$subdirectory1/final_air.txt" "$output\_air.txt"`;
	$sim1 = `cp "$subdirectory1/final.air" "$output\.air"`;
	$sim1 = `rm $subdirectory1/*`;
	$sim1 = `rmdir $subdirectory1`;


print "FINISHED...\n\n";
